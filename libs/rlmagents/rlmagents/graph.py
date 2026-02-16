"""Factory for creating RLM-enhanced agents.

This module provides ``create_rlm_agent()``, which implements a complete agent harness
with Aleph's RLM (Recursive Language Model) capabilities including context isolation,
evidence tracking, sandboxed Python REPL, and recipe pipelines.

The RLM agent includes planning, filesystem, sub-agents, plus:
- 23 RLM tools for structured analysis and evidence-backed reasoning
- Context isolation for large data analysis
- Auto-loading of large tool results into RLM contexts
- Cross-context search and analysis capabilities
- Integration with skills, memory, and summarization
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig, TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ResponseFormat
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.cache.base import BaseCache
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from rlmagents.middleware.rlm import RLMMiddleware

if TYPE_CHECKING:
    from rlmagents._standalone.backends.protocol import BackendFactory, BackendProtocol
    from rlmagents._standalone.middleware.subagents import CompiledSubAgent, SubAgent


# Import standalone deepagents functionality
from rlmagents._standalone.backends import StateBackend
from rlmagents._standalone.middleware.filesystem import FilesystemMiddleware
from rlmagents._standalone.middleware.memory import MemoryMiddleware
from rlmagents._standalone.middleware.patch_tool_calls import PatchToolCallsMiddleware
from rlmagents._standalone.middleware.skills import SkillsMiddleware
from rlmagents._standalone.middleware.subagents import (
    GENERAL_PURPOSE_SUBAGENT,
    CompiledSubAgent,
    SubAgent,
    SubAgentMiddleware,
)
from rlmagents._standalone.middleware.summarization import SummarizationMiddleware, _compute_summarization_defaults
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

# Load base prompt from incorporated deepagents code
BASE_AGENT_PROMPT = (Path(__file__).resolve().parent / "base_prompt.md").read_text()


def get_default_model() -> ChatAnthropic:
    """Get the default model for rlmagents.

    Returns:
        `ChatAnthropic` instance configured with Claude Sonnet 4.5.
    """
    return ChatAnthropic(
        model_name="claude-sonnet-4-5-20250929",
        max_tokens=20000,  # type: ignore[call-arg]
    )


def create_rlm_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    response_format: ResponseFormat | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
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
    """Create a deep agent enhanced with RLM context isolation and reasoning.

    This function extends ``create_deep_agent()`` by integrating RLM middleware
    with full support for skills, memory, summarization, and sub-agent RLM integration.

    The resulting agent has:
    - All DeepAgents tools (planning, filesystem, shell, sub-agents)
    - 22 RLM tools (context isolation, evidence tracking, Python REPL, recipes)
    - Skills middleware for domain-specific capabilities
    - Memory middleware for persistent AGENTS.md context
    - Summarization middleware for long conversation handling
    - Optional RLM tools in sub-agents for delegated analysis

    Args:
        model: The LLM to use. Defaults to claude-sonnet-4-5-20250929.

            Use the `provider:model` format (e.g., `openai:gpt-5`) to switch models.
        tools: Additional tools beyond the built-in ones.

            These are added to both the main agent and sub-agents (unless overridden).
        system_prompt: Custom system instructions prepended to the base prompt.

            If a string, concatenated with the base deep agent prompt.
        middleware: Additional middleware appended after the standard stack.

            Standard stack: RLMMiddleware, TodoListMiddleware, FilesystemMiddleware,
            SkillsMiddleware (if skills), MemoryMiddleware (if memory),
            SubAgentMiddleware, SummarizationMiddleware, AnthropicPromptCachingMiddleware,
            PatchToolCallsMiddleware, HumanInTheLoopMiddleware (if interrupt_on).
        subagents: Sub-agents for delegating work.

            Each sub-agent should be a dict with name, description, prompt, and optional
            tools, model, middleware, and skills.

            When `enable_rlm_in_subagents=True`, sub-agents automatically get RLM middleware
            unless explicitly disabled in their middleware list.
        skills: Optional list of skill source paths (e.g., `["/skills/user/", "/skills/project/"]`).

            Paths use POSIX conventions (forward slashes) and are relative to the backend's root.
            Later sources override earlier ones for skills with the same name (last one wins).

            Skills are loaded from SKILL.md files in each skill directory.
        memory: Optional list of memory file paths (AGENTS.md files) to load
            (e.g., `["/memory/AGENTS.md"]`).

            Display names are automatically derived from paths.
            Memory is loaded at agent startup and added to the system prompt.
        response_format: A structured output response format for the agent.
        context_schema: The schema of the deep agent context.
        checkpointer: Optional Checkpointer for persisting agent state between runs.
        store: Optional store for persistent storage (required if backend uses StoreBackend).
        backend: Optional backend for file storage and execution.

            Pass either a Backend instance or a callable factory like `lambda rt: StateBackend(rt)`.
            For execution support, use a backend that implements SandboxBackendProtocol.

            Defaults to StateBackend if not provided.
        interrupt_on: Mapping of tool names to interrupt configs for human-in-the-loop approval.

            Example: `interrupt_on={"edit_file": True}` pauses before every edit.
        debug: Whether to enable debug mode. Passed through to create_agent.
        name: The name of the agent. Passed through to create_agent.
        cache: The cache to use for the agent. Passed through to create_agent.
        sandbox_timeout: REPL sandbox timeout in seconds for RLM Python execution.

            Defaults to 180 seconds.
        context_policy: Context policy for RLM: "trusted" or "isolated".

            Defaults to "trusted".
        auto_load_threshold: Auto-load file results larger than this (chars) into RLM context.

            When a DeepAgents tool (e.g., read_file) returns content larger than this threshold,
            it is automatically loaded into an RLM context for efficient analysis.

            Defaults to 10,000 characters. Set to 0 to disable auto-loading.
        rlm_system_prompt: Override the default RLM workflow prompt.

            If None, uses the default prompt from rlm_prompt.md.
        enable_rlm_in_subagents: Whether to enable RLM middleware in sub-agents.

            When True (default), sub-agents get RLM context isolation and tools automatically.
            Set to False to use standard DeepAgents sub-agents without RLM features.

    Returns:
        A compiled LangGraph agent with RLM-enhanced capabilities.

        The agent has approximately 31+ tools:
        - 9 DeepAgents base tools (write_todos, ls, read_file, write_file, edit_file,
          glob, grep, execute, task)
        - 22 RLM tools (load_context, list_contexts, diff_contexts, save_session,
          load_session, peek_context, search_context, semantic_search, exec_python,
          get_variable, think, evaluate_progress, summarize_so_far, get_evidence,
          finalize, get_status, rlm_tasks, validate_recipe, estimate_recipe,
          run_recipe, run_recipe_code, configure_rlm)
        - Additional tools from skills and custom tools parameter

    Example:
        Basic usage with default settings:

        ```python
        from rlmagents import create_rlm_agent

        agent = create_rlm_agent()
        result = agent.invoke({"messages": [{"role": "user", "content": "Analyze this dataset..."}]})
        ```

        With skills and memory:

        ```python
        agent = create_rlm_agent(
            model="claude-sonnet-4-5-20250929",
            skills=["/skills/data-analysis/", "/skills/domain/finance/"],
            memory=["/memory/project-AGENTS.md"],
            auto_load_threshold=5000,  # Auto-load smaller files
        )
        ```

        With custom sub-agents (RLM-enabled by default):

        ```python
        from deepagents.middleware.subagents import SubAgent

        research_agent: SubAgent = {
            "name": "researcher",
            "description": "Conducts deep research on topics",
            "prompt": "You are a research specialist. Use RLM tools for structured analysis.",
            "skills": ["/skills/research/"],
        }

        agent = create_rlm_agent(
            subagents=[research_agent],
            enable_rlm_in_subagents=True,  # Default: sub-agents get RLM tools
        )
        ```

        With human-in-the-loop approval:

        ```python
        agent = create_rlm_agent(
            interrupt_on={
                "edit_file": True,
                "execute": True,
                "run_recipe": True,  # Approve recipe execution
            },
        )
        ```
    """
    # Resolve model
    if model is None:
        model = get_default_model()
    elif isinstance(model, str):
        if model.startswith("openai:"):
            model_init_params: dict = {"use_responses_api": True}
        else:
            model_init_params = {}
        model = init_chat_model(model, **model_init_params)

    # Compute summarization defaults based on model profile
    summarization_defaults = _compute_summarization_defaults(model)

    # Resolve backend
    resolved_backend = backend if backend is not None else StateBackend

    # Create RLM middleware
    rlm_mw = RLMMiddleware(
        sandbox_timeout=sandbox_timeout,
        context_policy=context_policy,
        auto_load_threshold=auto_load_threshold,
        system_prompt=rlm_system_prompt,
    )

    # Build main agent middleware stack
    deepagent_middleware: list[AgentMiddleware] = [
        rlm_mw,  # RLM middleware first for context isolation
        TodoListMiddleware(),
    ]

    # Add memory middleware if memory sources provided
    if memory is not None:
        deepagent_middleware.append(
            MemoryMiddleware(backend=resolved_backend, sources=memory)
        )

    # Add skills middleware if skills provided
    if skills is not None:
        deepagent_middleware.append(
            SkillsMiddleware(backend=resolved_backend, sources=skills)
        )

    # Add filesystem middleware
    deepagent_middleware.append(
        FilesystemMiddleware(backend=resolved_backend)
    )

    # Process sub-agents with optional RLM integration
    processed_subagents: list[SubAgent | CompiledSubAgent] = []

    for spec in subagents or []:
        if "runnable" in spec:
            # CompiledSubAgent - use as-is
            processed_subagents.append(spec)
        else:
            # SubAgent - fill in defaults and prepend base middleware
            subagent_model = spec.get("model", model)
            if isinstance(subagent_model, str):
                subagent_model = init_chat_model(subagent_model)

            # Build middleware: base stack + RLM (if enabled) + skills (if specified) + user's middleware
            subagent_summarization_defaults = _compute_summarization_defaults(subagent_model)
            subagent_middleware: list[AgentMiddleware] = [
                TodoListMiddleware(),
            ]

            # Add RLM middleware to sub-agent if enabled
            if enable_rlm_in_subagents:
                # Create a separate RLM middleware instance for the sub-agent
                # with its own session manager for context isolation
                subagent_rlm_mw = RLMMiddleware(
                    sandbox_timeout=sandbox_timeout,
                    context_policy=context_policy,
                    auto_load_threshold=auto_load_threshold,
                    system_prompt=rlm_system_prompt,
                )
                subagent_middleware.append(subagent_rlm_mw)

            subagent_middleware.extend([
                FilesystemMiddleware(backend=resolved_backend),
                SummarizationMiddleware(
                    model=subagent_model,
                    backend=resolved_backend,
                    trigger=subagent_summarization_defaults["trigger"],
                    keep=subagent_summarization_defaults["keep"],
                    trim_tokens_to_summarize=None,
                    truncate_args_settings=subagent_summarization_defaults["truncate_args_settings"],
                ),
                AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
                PatchToolCallsMiddleware(),
            ])

            # Add skills to sub-agent if specified
            subagent_skills = spec.get("skills")
            if subagent_skills:
                subagent_middleware.append(
                    SkillsMiddleware(backend=resolved_backend, sources=subagent_skills)
                )

            # Add user-provided middleware
            subagent_middleware.extend(spec.get("middleware", []))

            # Convert 'prompt' to 'system_prompt' for deepagents compatibility
            processed_spec: SubAgent = {
                **spec,
                "model": subagent_model,
                "tools": spec.get("tools", tools or []),
                "middleware": subagent_middleware,
            }
            # Ensure system_prompt is set (use 'prompt' if provided, otherwise use spec's system_prompt)
            if "prompt" in spec:
                processed_spec["system_prompt"] = spec["prompt"]
            elif "system_prompt" not in processed_spec:
                processed_spec["system_prompt"] = spec.get("description", "You are a helpful assistant.")
            
            processed_subagents.append(processed_spec)

    # Build general-purpose sub-agent with RLM-enhanced middleware stack
    gp_middleware: list[AgentMiddleware] = [
        TodoListMiddleware(),
    ]

    # Add RLM to general-purpose sub-agent if enabled
    if enable_rlm_in_subagents:
        gp_rlm_mw = RLMMiddleware(
            sandbox_timeout=sandbox_timeout,
            context_policy=context_policy,
            auto_load_threshold=auto_load_threshold,
            system_prompt=rlm_system_prompt,
        )
        gp_middleware.append(gp_rlm_mw)

    gp_middleware.extend([
        FilesystemMiddleware(backend=resolved_backend),
        SummarizationMiddleware(
            model=model,
            backend=resolved_backend,
            trigger=summarization_defaults["trigger"],
            keep=summarization_defaults["keep"],
            trim_tokens_to_summarize=None,
            truncate_args_settings=summarization_defaults["truncate_args_settings"],
        ),
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
        PatchToolCallsMiddleware(),
    ])

    if skills is not None:
        gp_middleware.append(SkillsMiddleware(backend=resolved_backend, sources=skills))

    if interrupt_on is not None:
        gp_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    general_purpose_spec: SubAgent = {
        **GENERAL_PURPOSE_SUBAGENT,
        "model": model,
        "tools": tools or [],
        "middleware": gp_middleware,
    }

    # Combine GP with processed user-provided sub-agents
    all_subagents: list[SubAgent | CompiledSubAgent] = [general_purpose_spec, *processed_subagents]

    # Add sub-agent middleware
    deepagent_middleware.append(
        SubAgentMiddleware(
            backend=resolved_backend,
            subagents=all_subagents,
        )
    )

    # Add summarization middleware
    deepagent_middleware.append(
        SummarizationMiddleware(
            model=model,
            backend=resolved_backend,
            trigger=summarization_defaults["trigger"],
            keep=summarization_defaults["keep"],
            trim_tokens_to_summarize=None,
            truncate_args_settings=summarization_defaults["truncate_args_settings"],
        )
    )

    # Add Anthropic prompt caching
    deepagent_middleware.append(
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore")
    )

    # Add patch tool calls middleware
    deepagent_middleware.append(PatchToolCallsMiddleware())

    # Add user-provided middleware
    if middleware:
        deepagent_middleware.extend(middleware)

    # Add human-in-the-loop middleware if interrupt_on provided
    if interrupt_on is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    # Combine system_prompt with BASE_AGENT_PROMPT
    if system_prompt is None:
        final_system_prompt: str | SystemMessage = BASE_AGENT_PROMPT
    elif isinstance(system_prompt, SystemMessage):
        # SystemMessage: append BASE_AGENT_PROMPT to content_blocks
        new_content = [
            *system_prompt.content_blocks,
            {"type": "text", "text": f"\n\n{BASE_AGENT_PROMPT}"},
        ]
        final_system_prompt = SystemMessage(content=new_content)
    else:
        # String: simple concatenation
        final_system_prompt = system_prompt + "\n\n" + BASE_AGENT_PROMPT

    return create_agent(
        model,
        system_prompt=final_system_prompt,
        tools=tools,
        middleware=deepagent_middleware,
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        backend=resolved_backend,
        debug=debug,
        name=name,
        cache=cache,
    ).with_config({"recursion_limit": 1000})
