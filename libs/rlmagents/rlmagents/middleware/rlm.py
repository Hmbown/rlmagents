"""RLMMiddleware -- RLM context isolation for the agent middleware stack."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import SystemMessage, ToolMessage
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
        )
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

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        self._sync_manager_loop()
        prompt = self._get_prompt()
        if prompt:
            new_system_message = _append_to_system_message(request.system_message, prompt)
            request = request.override(system_message=new_system_message)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        self._sync_manager_loop()
        prompt = self._get_prompt()
        if prompt:
            new_system_message = _append_to_system_message(request.system_message, prompt)
            request = request.override(system_message=new_system_message)
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
