"""Factory for creating RLM-enhanced agents.

This module provides ``create_rlm_agent()``, a complete standalone agent harness with:
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

import inspect
import re
from collections.abc import Callable, Sequence
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as _get_package_version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
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

if TYPE_CHECKING:
    from rlmagents._harness.backends.protocol import BackendFactory, BackendProtocol
    from rlmagents._harness.middleware.subagents import CompiledSubAgent, SubAgent

# Import incorporated harness
from rlmagents._harness.backends import StateBackend
from rlmagents._harness.middleware.filesystem import FilesystemMiddleware
from rlmagents._harness.middleware.memory import MemoryMiddleware
from rlmagents._harness.middleware.patch_tool_calls import PatchToolCallsMiddleware
from rlmagents._harness.middleware.skills import SkillsMiddleware
from rlmagents._harness.middleware.subagents import (
    GENERAL_PURPOSE_SUBAGENT,
    CompiledSubAgent,
    SubAgent,
    SubAgentMiddleware,
)
from rlmagents._harness.middleware.summarization import (
    SummarizationMiddleware,
    _compute_summarization_defaults,
)
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware

# Load base prompt
BASE_AGENT_PROMPT = (Path(__file__).resolve().parent / "base_prompt.md").read_text()

_LANGCHAIN_MIN_VERSION = (1, 2, 10)
_LANGCHAIN_MAX_VERSION = (1, 3, 0)
_CODING_RESPONSE_FIELDS = ("plan", "edits", "verification", "risks", "confidence")
_CODING_RESPONSE_FIELD_DESCRIPTIONS = {
    "plan": "Execution plan for requested coding work.",
    "edits": "Concrete file- and line-level edits to apply.",
    "verification": "Commands or checks to verify correctness.",
    "risks": "Caveats and likely regressions.",
    "confidence": "Confidence level from 0 to 1.",
}


def _get_langchain_version() -> str:
    """Return the installed langchain version."""
    return _get_package_version("langchain")


def _parse_version_tuple(version: str) -> tuple[int, int, int]:
    match = re.match(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)", version)
    if not match:
        raise ValueError(f"Unable to parse langchain version string: {version!r}")
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
    )


def _is_langchain_version_supported(version: str) -> bool:
    parsed = _parse_version_tuple(version)
    return _LANGCHAIN_MIN_VERSION <= parsed < _LANGCHAIN_MAX_VERSION


def _extract_system_prompt_text(system_prompt: str | SystemMessage | None) -> str:
    if system_prompt is None:
        return ""
    if isinstance(system_prompt, SystemMessage):
        content = system_prompt.content
        if isinstance(content, str):
            return content
        return "".join(
            (
                block
                if isinstance(block, str)
                else str(block.get("text", ""))
                if isinstance(block, dict)
                else str(block)
            )
            for block in content
        )
    return str(system_prompt)


def _is_coding_task(system_prompt: str | SystemMessage | None) -> bool:
    prompt = _extract_system_prompt_text(system_prompt).lower()
    return bool(
        re.search(
            r"\b(code|coding|implement|refactor|patch|bug|function|class)\b",
            prompt,
        )
    )


def _coding_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": _CODING_RESPONSE_FIELD_DESCRIPTIONS["plan"],
                },
                "edits": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": _CODING_RESPONSE_FIELD_DESCRIPTIONS["edits"],
                },
                "verification": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": _CODING_RESPONSE_FIELD_DESCRIPTIONS["verification"],
                },
                "risks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": _CODING_RESPONSE_FIELD_DESCRIPTIONS["risks"],
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": _CODING_RESPONSE_FIELD_DESCRIPTIONS["confidence"],
                },
            },
            "required": list(_CODING_RESPONSE_FIELDS),
            "additionalProperties": False,
        },
    }


@lru_cache(maxsize=1)
def _assert_langchain_compatibility() -> None:
    """Fail fast if the installed langchain API is unsupported."""
    try:
        version = _get_langchain_version()
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "rlmagents requires langchain to build agents."
        ) from exc

    if not _is_langchain_version_supported(version):
        minimum_version = ".".join(map(str, _LANGCHAIN_MIN_VERSION))
        maximum_version = ".".join(map(str, _LANGCHAIN_MAX_VERSION))
        raise RuntimeError(
            "rlmagents supports langchain version >= "
            f"{minimum_version} and < {maximum_version}; "
            f"installed version is {version}."
        )

    create_agent_signature = inspect.signature(create_agent)
    parameters = create_agent_signature.parameters
    if "backend" not in parameters and not (
        "checkpointer" in parameters and "store" in parameters
    ):
        raise RuntimeError(
            "Installed langchain.agents.create_agent is missing required persistence args. "
            "Expected `backend` or both `checkpointer` and `store`."
        )


def _build_create_agent_kwargs(
    checkpointer: Checkpointer | None,
    store: BaseStore | None,
    backend: BackendProtocol | BackendFactory | None,
    debug: bool,
    name: str | None,
    cache: BaseCache | None,
) -> dict[str, Any]:
    """Build a kwargs mapping for langchain.agents.create_agent."""
    create_agent_parameters = inspect.signature(create_agent).parameters
    kwargs: dict[str, Any] = {}

    if "backend" in create_agent_parameters:
        kwargs["backend"] = backend
    else:
        kwargs["checkpointer"] = checkpointer
        kwargs["store"] = store

    if "debug" in create_agent_parameters:
        kwargs["debug"] = debug
    if "name" in create_agent_parameters:
        kwargs["name"] = name
    if "cache" in create_agent_parameters:
        kwargs["cache"] = cache

    return kwargs


def get_default_model() -> ChatAnthropic:
    """Get the default model for rlmagents."""
    return ChatAnthropic(
        model_name="claude-sonnet-4-5-20250929",
        max_tokens=20000,
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
    """Create an RLM-enhanced agent.

    This is a complete standalone agent harness with planning, filesystem,
    sub-agents, plus 23 RLM tools for context isolation and evidence tracking.

    Args:
        model: The LLM to use. Defaults to claude-sonnet-4-5-20250929.
        tools: Additional tools beyond the built-in ones.
        system_prompt: Custom system instructions.
        middleware: Additional middleware to add.
        subagents: Sub-agent specifications.
        skills: Skill source paths.
        memory: Memory file paths (AGENTS.md).
        response_format: Structured output format.
        context_schema: Context schema.
        checkpointer: Checkpointer for persistence.
        store: Store for persistent storage.
        backend: Backend for file storage/execution.
        interrupt_on: Tool interrupt configurations.
        debug: Enable debug mode.
        name: Agent name.
        cache: Cache for the agent.
        sandbox_timeout: RLM REPL timeout.
        context_policy: RLM context policy.
        auto_load_threshold: Auto-load threshold for large results.
        rlm_system_prompt: Custom RLM workflow prompt.
        enable_rlm_in_subagents: Enable RLM in sub-agents.

    Returns:
        Compiled LangGraph agent.
    """
    _assert_langchain_compatibility()

    # Resolve model
    if model is None:
        model = get_default_model()
    elif isinstance(model, str):
        if model.startswith("openai:"):
            model = init_chat_model(model, use_responses_api=True)
        else:
            model = init_chat_model(model)

    # Compute summarization defaults
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
    agent_middleware: list[AgentMiddleware] = [
        rlm_mw,
        TodoListMiddleware(),
    ]

    # Add memory if provided
    if memory is not None:
        agent_middleware.append(
            MemoryMiddleware(backend=resolved_backend, sources=memory)
        )

    # Add skills if provided
    if skills is not None:
        agent_middleware.append(
            SkillsMiddleware(backend=resolved_backend, sources=skills)
        )

    # Add filesystem
    agent_middleware.append(
        FilesystemMiddleware(backend=resolved_backend)
    )

    # Process sub-agents with RLM integration
    processed_subagents: list[SubAgent | CompiledSubAgent] = []
    for spec in subagents or []:
        if "runnable" in spec:
            processed_subagents.append(spec)
        else:
            subagent_model = spec.get("model", model)
            if isinstance(subagent_model, str):
                subagent_model = init_chat_model(subagent_model)

            subagent_summarization_defaults = _compute_summarization_defaults(subagent_model)
            subagent_middleware: list[AgentMiddleware] = [
                TodoListMiddleware(),
            ]

            if enable_rlm_in_subagents:
                subagent_middleware.append(
                    RLMMiddleware(
                        sandbox_timeout=sandbox_timeout,
                        context_policy=context_policy,
                        auto_load_threshold=auto_load_threshold,
                        system_prompt=rlm_system_prompt,
                    )
                )

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

            if spec.get("skills"):
                subagent_middleware.append(
                    SkillsMiddleware(backend=resolved_backend, sources=spec["skills"])
                )
            subagent_middleware.extend(spec.get("middleware", []))

            processed_spec: SubAgent = {
                **spec,
                "model": subagent_model,
                "tools": spec.get("tools", tools or []),
                "middleware": subagent_middleware,
            }
            if "prompt" in spec:
                processed_spec["system_prompt"] = spec["prompt"]
            elif "system_prompt" not in processed_spec:
                processed_spec["system_prompt"] = spec.get("description", "You are a helpful assistant.")
            processed_subagents.append(processed_spec)

    # Build general-purpose sub-agent
    gp_middleware: list[AgentMiddleware] = [
        TodoListMiddleware(),
    ]
    if enable_rlm_in_subagents:
        gp_middleware.append(
            RLMMiddleware(
                sandbox_timeout=sandbox_timeout,
                context_policy=context_policy,
                auto_load_threshold=auto_load_threshold,
                system_prompt=rlm_system_prompt,
            )
        )
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
    if skills:
        gp_middleware.append(SkillsMiddleware(backend=resolved_backend, sources=skills))
    if interrupt_on:
        gp_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    general_purpose_spec: SubAgent = {
        **GENERAL_PURPOSE_SUBAGENT,
        "model": model,
        "tools": tools or [],
        "middleware": gp_middleware,
    }

    all_subagents: list[SubAgent | CompiledSubAgent] = [general_purpose_spec, *processed_subagents]

    # Add sub-agent middleware
    agent_middleware.append(
        SubAgentMiddleware(
            backend=resolved_backend,
            subagents=all_subagents,
        )
    )

    # Add summarization
    agent_middleware.append(
        SummarizationMiddleware(
            model=model,
            backend=resolved_backend,
            trigger=summarization_defaults["trigger"],
            keep=summarization_defaults["keep"],
            trim_tokens_to_summarize=None,
            truncate_args_settings=summarization_defaults["truncate_args_settings"],
        )
    )

    # Add Anthropic caching
    agent_middleware.append(
        AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore")
    )

    # Add patch tool calls
    agent_middleware.append(PatchToolCallsMiddleware())

    # Add user middleware
    if middleware:
        agent_middleware.extend(middleware)

    # Add human-in-the-loop
    if interrupt_on is not None:
        agent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    # Combine system prompt
    if system_prompt is None:
        final_prompt: str | SystemMessage = BASE_AGENT_PROMPT
    elif isinstance(system_prompt, SystemMessage):
        new_content = [
            *system_prompt.content_blocks,
            {"type": "text", "text": f"\n\n{BASE_AGENT_PROMPT}"},
        ]
        final_prompt = SystemMessage(content=new_content)
    else:
        final_prompt = system_prompt + "\n\n" + BASE_AGENT_PROMPT

    if response_format is None and _is_coding_task(final_prompt):
        response_format = _coding_response_format()

    agent_kwargs = _build_create_agent_kwargs(
        checkpointer=checkpointer,
        store=store,
        backend=resolved_backend,
        debug=debug,
        name=name,
        cache=cache,
    )

    return create_agent(
        model,
        system_prompt=final_prompt,
        tools=tools,
        middleware=agent_middleware,
        response_format=response_format,
        context_schema=context_schema,
        **agent_kwargs,
    ).with_config({"recursion_limit": 1000})
