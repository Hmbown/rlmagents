"""RLMMiddleware -- RLM context isolation for the agent middleware stack."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from rlmagents.middleware._prompt import load_rlm_prompt
from rlmagents.middleware._state import RLMState
from rlmagents.middleware._tools import (
    DEFAULT_RLM_TOOL_PROFILE,
    RLM_TOOL_NAMES,
    RLMToolProfile,
    _build_rlm_tools,
)
from rlmagents.repl.sandbox import SandboxConfig
from rlmagents.session_manager import RLMSessionManager

_CLI_FILE_MENTION_METADATA_TAG = "RLMAGENTS_FILE_MENTIONS_V1"
_CLI_FILE_MENTION_METADATA_PATTERN = re.compile(
    rf"<{_CLI_FILE_MENTION_METADATA_TAG}>(?P<payload>.*?)</{_CLI_FILE_MENTION_METADATA_TAG}>",
    re.DOTALL,
)


def _append_to_system_message(
    system_message: SystemMessage | None,
    text: str,
) -> SystemMessage:
    """Append text to a system message, creating one if needed."""
    new_content: list[str | dict[str, str]] = (
        list(system_message.content_blocks) if system_message else []
    )
    if new_content:
        text = f"\n\n{text}"
    new_content.append({"type": "text", "text": text})
    return SystemMessage(content=new_content)


def _normalize_context_stem(path: Path) -> str:
    stem = path.stem or path.name or "file"
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", stem).strip("_").lower()
    return normalized or "file"


def _context_id_for_file(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    return f"mention_{_normalize_context_stem(path)}_{digest}"


def _message_is_user(message: object) -> bool:
    if isinstance(message, HumanMessage):
        return True
    if isinstance(message, dict):
        role = message.get("role")
        return isinstance(role, str) and role.lower() in {"user", "human"}
    message_type = getattr(message, "type", None)
    return isinstance(message_type, str) and message_type == "human"


def _message_content(message: object) -> object:
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", "")


def _text_from_message_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    text_blocks: list[str] = []
    for block in content:
        if isinstance(block, str):
            text_blocks.append(block)
            continue
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str):
            text_blocks.append(text)
    return "\n".join(text_blocks)


def _extract_cli_file_paths(message_text: str) -> list[str]:
    paths: list[str] = []
    for match in _CLI_FILE_MENTION_METADATA_PATTERN.finditer(message_text):
        payload = match.group("payload").strip()
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        raw_paths = parsed.get("paths")
        if not isinstance(raw_paths, list):
            continue
        for raw_path in raw_paths:
            if isinstance(raw_path, str) and raw_path:
                paths.append(raw_path)

    # Preserve order while removing duplicates.
    deduped: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path in seen:
            continue
        deduped.append(path)
        seen.add(path)
    return deduped


# Backwards-compatible export for tests and integrations that import this symbol.
_RLM_TOOL_NAMES = RLM_TOOL_NAMES


class RLMMiddleware(AgentMiddleware):
    """Middleware that adds RLM context isolation to the agent stack.

    Provides profile-driven tools for context loading, search, Python
    execution, evidence tracking, reasoning workflow, and recipe pipelines.

    When ``sub_query_model`` is provided, the ``sub_query()`` /
    ``llm_query()`` function inside ``exec_python`` will invoke that
    model — enabling the core recursive mechanism from Algorithm 1
    of the RLM paper.
    """

    state_schema = RLMState

    def __init__(
        self,
        *,
        sandbox_timeout: float = 180.0,
        context_policy: str = "trusted",
        hist_max_entries: int = 64,
        hist_max_code_chars: int = 280,
        hist_max_text_chars: int = 200,
        enable_final_sentinel: bool = False,
        inject_exec_metadata: bool = True,
        inject_exec_metadata_max_entries: int = 4,
        inject_exec_metadata_max_chars: int = 1200,
        auto_load_threshold: int = 10_000,
        auto_load_preview_chars: int = 600,
        tool_profile: RLMToolProfile = DEFAULT_RLM_TOOL_PROFILE,
        include_tools: Sequence[str] = (),
        exclude_tools: Sequence[str] = (),
        system_prompt: str | None = None,
        sub_query_model: object | None = None,
        sub_query_timeout: float = 120.0,
    ) -> None:
        self._manager = RLMSessionManager(
            sandbox_config=SandboxConfig(timeout_seconds=sandbox_timeout),
            context_policy=context_policy,
            sub_query_model=sub_query_model,  # type: ignore[arg-type]
            sub_query_timeout=sub_query_timeout,
            hist_max_entries=hist_max_entries,
            hist_max_code_chars=hist_max_code_chars,
            hist_max_text_chars=hist_max_text_chars,
            enable_final_sentinel=enable_final_sentinel,
        )
        self._inject_exec_metadata = inject_exec_metadata
        self._inject_exec_metadata_max_entries = max(1, inject_exec_metadata_max_entries)
        self._inject_exec_metadata_max_chars = max(200, inject_exec_metadata_max_chars)
        self._auto_load_threshold = auto_load_threshold
        self._auto_load_preview_chars = max(auto_load_preview_chars, 0)
        self._custom_prompt = system_prompt
        self.tools: Sequence[BaseTool] = _build_rlm_tools(
            self._manager,
            profile=tool_profile,
            include_tools=include_tools,
            exclude_tools=exclude_tools,
        )
        self._rlm_tool_names = frozenset(tool.name for tool in self.tools)

    @property
    def manager(self) -> RLMSessionManager:
        """Access the underlying session manager."""
        return self._manager

    # -- System prompt injection --------------------------------------------

    def _get_prompt(self) -> str:
        if self._custom_prompt is not None:
            return self._custom_prompt
        return load_rlm_prompt()

    def _sync_manager_loop(self) -> None:
        """Keep the manager's async bridge loop aligned with runtime."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._manager.set_loop(loop)

    def _exec_metadata_note(self) -> str | None:
        if not self._inject_exec_metadata:
            return None
        note = self._manager.format_recent_exec_metadata(
            limit=self._inject_exec_metadata_max_entries,
            max_chars=self._inject_exec_metadata_max_chars,
        )
        return note or None

    def _auto_load_cli_file_mentions(self, request: ModelRequest) -> str | None:
        """Load CLI `@file` mentions into RLM contexts for the latest user turn."""
        if not request.messages:
            return None

        latest_message = request.messages[-1]
        if not _message_is_user(latest_message):
            return None

        text_content = _text_from_message_content(_message_content(latest_message))
        if not text_content:
            return None

        file_paths = _extract_cli_file_paths(text_content)
        if not file_paths:
            return None

        loaded: list[str] = []
        failed: list[str] = []

        for raw_path in file_paths:
            file_path = Path(raw_path).expanduser()
            try:
                resolved_path = file_path.resolve(strict=True)
            except (FileNotFoundError, OSError) as exc:
                failed.append(f"{raw_path} ({exc})")
                continue

            if not resolved_path.is_file():
                failed.append(f"{resolved_path} (not a file)")
                continue

            try:
                content = resolved_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                failed.append(f"{resolved_path} (non-UTF-8 text)")
                continue
            except OSError as exc:
                failed.append(f"{resolved_path} ({exc})")
                continue

            context_id = _context_id_for_file(resolved_path)
            self._manager.create_session(
                content,
                context_id=context_id,
                format_hint="auto",
                line_number_base=1,
            )
            loaded.append(f"{context_id} <- {resolved_path}")

        if not loaded and not failed:
            return None

        lines = ["[CLI @file auto-load]"]
        if loaded:
            lines.append(f"Loaded contexts: {'; '.join(loaded)}")
            lines.append("Use search_context/peek_context on these context IDs.")
        if failed:
            lines.append(f"Skipped: {'; '.join(failed)}")
        return "\n".join(lines)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        self._sync_manager_loop()
        system_message = request.system_message
        prompt = self._get_prompt()
        if prompt:
            system_message = _append_to_system_message(system_message, prompt)
        auto_load_note = self._auto_load_cli_file_mentions(request)
        if auto_load_note:
            system_message = _append_to_system_message(system_message, auto_load_note)
        exec_metadata_note = self._exec_metadata_note()
        if exec_metadata_note:
            system_message = _append_to_system_message(system_message, exec_metadata_note)
        if system_message is not request.system_message:
            request = request.override(system_message=system_message)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        self._sync_manager_loop()
        system_message = request.system_message
        prompt = self._get_prompt()
        if prompt:
            system_message = _append_to_system_message(system_message, prompt)
        auto_load_note = self._auto_load_cli_file_mentions(request)
        if auto_load_note:
            system_message = _append_to_system_message(system_message, auto_load_note)
        exec_metadata_note = self._exec_metadata_note()
        if exec_metadata_note:
            system_message = _append_to_system_message(system_message, exec_metadata_note)
        if system_message is not request.system_message:
            request = request.override(system_message=system_message)
        return await handler(request)

    # -- Auto-load large tool results into RLM context ----------------------

    def _maybe_auto_load(
        self, result: ToolMessage | Command, tool_name: str
    ) -> ToolMessage | Command:
        """If a tool result is large, auto-load it into an RLM context."""
        if self._auto_load_threshold <= 0:
            return result
        if tool_name in self._rlm_tool_names:
            return result
        if not isinstance(result, ToolMessage):
            return result

        content = result.content
        if isinstance(content, str) and len(content) > self._auto_load_threshold:
            # Generate a context name from the tool name
            ctx_name = f"auto_{tool_name}"
            self._manager.create_session(content, context_id=ctx_name)
            hint = (
                f"[Large result ({len(content):,} chars) auto-loaded into RLM context "
                f"'{ctx_name}'. Use search_context/peek_context to explore.]"
            )
            preview = ""
            if self._auto_load_preview_chars > 0:
                preview = content[: self._auto_load_preview_chars]
                if len(content) > self._auto_load_preview_chars:
                    preview += (
                        f"\n... [preview truncated at {self._auto_load_preview_chars:,} chars]"
                    )
            response_content = hint if not preview else f"{preview}\n\n{hint}"
            return ToolMessage(
                content=response_content,
                tool_call_id=result.tool_call_id,
                name=result.name,
            )
        return result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        self._sync_manager_loop()
        tool_result = handler(request)
        return self._maybe_auto_load(tool_result, request.tool_call.get("name", ""))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        self._sync_manager_loop()
        tool_result = await handler(request)
        return self._maybe_auto_load(tool_result, request.tool_call.get("name", ""))
