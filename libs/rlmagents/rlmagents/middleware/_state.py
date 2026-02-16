"""RLM state schema for LangGraph checkpointing."""

from __future__ import annotations

from typing import NotRequired

from langchain.agents.middleware.types import AgentState


class RLMState(AgentState):
    """Extends AgentState with lightweight RLM metadata.

    Heavyweight data (REPL contexts, evidence) lives in RLMSessionManager
    as an instance attribute on the middleware, not in LangGraph state.
    """

    rlm_context_ids: NotRequired[list[str]]
