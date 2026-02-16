"""RLMMiddleware -- brings Aleph's RLM capabilities into DeepAgents' middleware stack."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from aleph.repl.sandbox import SandboxConfig

from rlmagents.middleware._prompt import load_rlm_prompt
from rlmagents.middleware._state import RLMState
from rlmagents.middleware._tools import _build_rlm_tools
from rlmagents.session_manager import RLMSessionManager

try:
    from deepagents.middleware._utils import append_to_system_message
except ImportError:  # pragma: no cover
    # Fallback if deepagents isn't installed in the current env
    from langchain_core.messages import SystemMessage

    def append_to_system_message(  # type: ignore[misc]
        system_message: SystemMessage | None, text: str,
    ) -> SystemMessage:
        new_content: list[str | dict[str, str]] = (
            list(system_message.content_blocks) if system_message else []
        )
        if new_content:
            text = f"\n\n{text}"
        new_content.append({"type": "text", "text": text})
        return SystemMessage(content=new_content)


# Tools that should NOT trigger auto-load interception
_RLM_TOOL_NAMES = frozenset({
    "load_context", "list_contexts", "diff_contexts", "save_session", "load_session",
    "peek_context", "search_context", "semantic_search", "exec_python", "get_variable",
    "think", "evaluate_progress", "summarize_so_far", "get_evidence", "finalize",
    "get_status", "rlm_tasks",
    "validate_recipe", "estimate_recipe", "run_recipe", "run_recipe_code",
    "configure_rlm",
})


class RLMMiddleware(AgentMiddleware):
    """Middleware that adds Aleph's RLM context isolation to DeepAgents.

    Provides 22 tools for context loading, search, Python execution,
    evidence tracking, reasoning workflow, and recipe pipelines.
    """

    state_schema = RLMState

    def __init__(
        self,
        *,
        sandbox_timeout: float = 180.0,
        context_policy: str = "trusted",
        auto_load_threshold: int = 10_000,
        system_prompt: str | None = None,
    ) -> None:
        self._manager = RLMSessionManager(
            sandbox_config=SandboxConfig(timeout_seconds=sandbox_timeout),
            context_policy=context_policy,
        )
        self._auto_load_threshold = auto_load_threshold
        self._custom_prompt = system_prompt
        self.tools: Sequence[BaseTool] = _build_rlm_tools(self._manager)

    @property
    def manager(self) -> RLMSessionManager:
        """Access the underlying session manager."""
        return self._manager

    # -- System prompt injection --------------------------------------------

    def _get_prompt(self) -> str:
        if self._custom_prompt is not None:
            return self._custom_prompt
        return load_rlm_prompt()

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        prompt = self._get_prompt()
        if prompt:
            new_system_message = append_to_system_message(request.system_message, prompt)
            request = request.override(system_message=new_system_message)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        prompt = self._get_prompt()
        if prompt:
            new_system_message = append_to_system_message(request.system_message, prompt)
            request = request.override(system_message=new_system_message)
        return await handler(request)

    # -- Auto-load large tool results into RLM context ----------------------

    def _maybe_auto_load(self, result: ToolMessage | Command, tool_name: str) -> ToolMessage | Command:
        """If a tool result is large, auto-load it into an RLM context."""
        if self._auto_load_threshold <= 0:
            return result
        if tool_name in _RLM_TOOL_NAMES:
            return result
        if not isinstance(result, ToolMessage):
            return result

        content = result.content
        if isinstance(content, str) and len(content) > self._auto_load_threshold:
            # Generate a context name from the tool name
            ctx_name = f"auto_{tool_name}"
            self._manager.create_session(content, context_id=ctx_name)
            hint = (
                f"\n\n[Large result ({len(content):,} chars) auto-loaded into RLM context "
                f"'{ctx_name}'. Use search_context/peek_context to explore.]"
            )
            return ToolMessage(
                content=content[:2000] + hint,
                tool_call_id=result.tool_call_id,
                name=result.name,
            )
        return result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        tool_result = handler(request)
        return self._maybe_auto_load(tool_result, request.tool_call.get("name", ""))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        tool_result = await handler(request)
        return self._maybe_auto_load(tool_result, request.tool_call.get("name", ""))
