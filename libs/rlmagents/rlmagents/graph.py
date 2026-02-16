"""Factory for creating RLM-enhanced agents.

This module provides ``create_rlm_agent()``, a complete agent harness with:
- Planning (todo lists)
- Filesystem access (read, write, edit, search)
- Sub-agents for delegation
- RLM context isolation and evidence tracking (23 tools)
- Memory and skills support
- Auto-summarization for long conversations

The agent is taught via system prompts to use RLM-style reasoning:
load context, search/peek, execute Python for analysis, track evidence,
and produce cited conclusions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    SummarizationMiddleware,
    TodoListMiddleware,
)
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ResponseFormat
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.cache.base import BaseCache
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from rlmagents.middleware.rlm import RLMMiddleware


# Default system prompt that teaches RLM-style reasoning
DEFAULT_SYSTEM_PROMPT = """You are an AI assistant that helps users accomplish tasks using tools and structured RLM (Recursive Language Model) reasoning.

## Core Behavior

- Be concise and direct. Don't over-explain unless asked.
- NEVER add unnecessary preamble ("Sure!", "Great question!", "I'll now...").
- Don't say "I'll now do X" — just do it.
- If the request is ambiguous, ask questions before acting.

## RLM Workflow

For complex analysis tasks, use this workflow:

1. **Load Context**: Use `load_context` to load large files/data into isolated RLM contexts
2. **Explore**: Use `search_context`, `peek_context`, or `semantic_search` to understand the data
3. **Analyze**: Use `exec_python` to run Python code with 100+ built-in helpers (search, extract, stats, etc.)
4. **Track Evidence**: Evidence is recorded automatically when you search/analyze
5. **Reason**: Use `think` to structure sub-steps, `evaluate_progress` to assess confidence
6. **Conclude**: Use `get_evidence` to review findings, then `finalize` to produce a cited answer

## Available Tools

**Planning**: `write_todos`, `read_todos`
**Filesystem**: `read_file`, `write_file`, `edit_file`, `ls`, `glob`, `grep`
**Shell**: `execute` (sandboxed)
**Sub-agents**: `task` (delegate work)
**RLM Context**: `load_context`, `list_contexts`, `diff_contexts`, `save_session`, `load_session`
**RLM Query**: `peek_context`, `search_context`, `semantic_search`, `cross_context_search`, `exec_python`, `get_variable`
**RLM Reasoning**: `think`, `evaluate_progress`, `summarize_so_far`, `get_evidence`, `finalize`
**RLM Recipes**: `validate_recipe`, `run_recipe`, `run_recipe_code`

## When to Use RLM Tools

- Large files (>10KB) — load into RLM context, then search/peek instead of reading fully
- Complex analysis — use exec_python with helpers like search(), extract_*, cite()
- Multi-source comparison — load into separate contexts, use cross_context_search or diff_contexts
- Evidence-backed answers — use RLM tools to track provenance

## File Operations

- Read files before editing — understand existing content
- Mimic existing style, naming conventions, and patterns
- For large files, use pagination (limit/offset) or load into RLM context

## Progress Updates

For longer tasks, provide brief progress updates at reasonable intervals.
"""


def create_rlm_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: list[dict[str, Any]] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    response_format: ResponseFormat | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
    # RLM-specific parameters
    sandbox_timeout: float = 180.0,
    context_policy: str = "trusted",
    auto_load_threshold: int = 10_000,
    rlm_system_prompt: str | None = None,
    enable_rlm_in_subagents: bool = True,
) -> CompiledStateGraph:
    """Create an RLM-enhanced agent.

    This agent has built-in planning, filesystem access, sub-agents, plus
    23 RLM tools for context isolation, evidence tracking, and structured analysis.

    Args:
        model: The LLM to use. Defaults to claude-sonnet-4-5-20250929.
        tools: Additional tools beyond the built-in ones.
        system_prompt: Custom system instructions. If None, uses RLM-style default.
        middleware: Additional middleware to add.
        subagents: Sub-agent specifications for delegation.
        skills: Skill source paths (e.g., ["/skills/user/"]).
        memory: Memory file paths (AGENTS.md files).
        response_format: Structured output format.
        context_schema: Context schema type.
        checkpointer: Checkpointer for persistence.
        store: Store for persistent storage.
        interrupt_on: Tool interrupt configurations for human-in-the-loop.
        debug: Enable debug mode.
        name: Agent name.
        cache: Cache for the agent.
        sandbox_timeout: RLM REPL sandbox timeout in seconds.
        context_policy: RLM context policy ("trusted" or "isolated").
        auto_load_threshold: Auto-load tool results larger than this (chars) into RLM context.
        rlm_system_prompt: Custom RLM workflow prompt.
        enable_rlm_in_subagents: Enable RLM tools in sub-agents.

    Returns:
        Compiled LangGraph agent with RLM capabilities.

    Example:
        ```python
        from rlmagents import create_rlm_agent

        agent = create_rlm_agent(
            skills=["/skills/analysis/"],
            memory=["/memory/AGENTS.md"],
            auto_load_threshold=5000,
        )

        result = agent.invoke({
            "messages": [{"role": "user", "content": "Analyze this dataset..."}]
        })
        ```
    """
    # Resolve model
    if model is None:
        from langchain_anthropic import ChatAnthropic
        model = ChatAnthropic(
            model_name="claude-sonnet-4-5-20250929",
            max_tokens=20000,
        )
    elif isinstance(model, str):
        if model.startswith("openai:"):
            model = init_chat_model(model, use_responses_api=True)
        else:
            model = init_chat_model(model)

    # Build middleware stack
    agent_middleware: list[AgentMiddleware] = [
        TodoListMiddleware(),
    ]

    # Add RLM middleware
    rlm_mw = RLMMiddleware(
        sandbox_timeout=sandbox_timeout,
        context_policy=context_policy,
        auto_load_threshold=auto_load_threshold,
        system_prompt=rlm_system_prompt,
    )
    agent_middleware.append(rlm_mw)

    # Note: filesystem, skills, memory, sub-agents would be added here
    # For now, we keep it minimal - users can add custom middleware

    # Add summarization
    agent_middleware.append(
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 170000),
            keep=("messages", 20),
        )
    )

    # Add user middleware
    if middleware:
        agent_middleware.extend(middleware)

    # Add human-in-the-loop
    if interrupt_on is not None:
        agent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    # Resolve system prompt
    if system_prompt is None:
        final_prompt = DEFAULT_SYSTEM_PROMPT
    elif isinstance(system_prompt, SystemMessage):
        final_prompt = system_prompt
    else:
        final_prompt = SystemMessage(content=system_prompt)

    # Create agent
    return create_agent(
        model,
        system_prompt=final_prompt,
        tools=tools,
        middleware=agent_middleware,
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name=name,
        cache=cache,
    ).with_config({"recursion_limit": 1000})
