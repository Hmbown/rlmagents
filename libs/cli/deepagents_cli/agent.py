"""Agent management and creation for the CLI."""

import os
import shutil
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from rlmagents import create_rlm_agent
from rlmagents._harness.backends import CompositeBackend  # noqa: PLC2701
from rlmagents._harness.backends.filesystem import FilesystemBackend  # noqa: PLC2701
from rlmagents._harness.backends.protocol import SandboxBackendProtocol
from rlmagents._harness.memory import MemoryMiddleware  # noqa: PLC2701
from rlmagents._harness.skills import SkillsMiddleware  # noqa: PLC2701

from deepagents_cli.backends import CLIShellBackend, patch_filesystem_middleware

if TYPE_CHECKING:
    from rlmagents._harness.subagents import (
        CompiledSubAgent,
        SubAgent,
    )
from langchain.agents.middleware import (
    InterruptOnConfig,
)
from langchain.agents.middleware.types import AgentState
from langchain.messages import ToolCall
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.pregel import Pregel
from langgraph.runtime import Runtime

from deepagents_cli.config import (
    COLORS,
    config,
    console,
    get_default_coding_instructions,
    get_glyphs,
    settings,
)
from deepagents_cli.integrations.sandbox_factory import get_default_working_dir
from deepagents_cli.local_context import LocalContextMiddleware
from deepagents_cli.subagents import list_subagents

DEFAULT_AGENT_NAME = "agent"
"""The default agent name used when no `-a` flag is provided."""

HarnessType = Literal["rlmagents", "deepagents"]

# Backward-compatibility alias for tests and integration hooks that still patch
# `create_deep_agent`. In rlmagents-cli, both harness names resolve to the
# rlmagents implementation.
create_deep_agent = create_rlm_agent


def list_agents() -> None:
    """List all available agents."""
    agent_dirs = settings.list_agent_dirs()
    if not agent_dirs:
        console.print("[yellow]No agents found.[/yellow]")
        console.print(
            "[dim]Agents will be created in ~/.rlmagents/ "
            "when you first use them.[/dim]",
            style=COLORS["dim"],
        )
        return

    console.print("\n[bold]Available Agents:[/bold]\n", style=COLORS["primary"])

    for agent_name in sorted(agent_dirs):
        agent_path = agent_dirs[agent_name]
        agent_md = agent_path / "AGENTS.md"
        is_default = agent_name == DEFAULT_AGENT_NAME
        default_label = " [dim](default)[/dim]" if is_default else ""

        bullet = get_glyphs().bullet
        if agent_md.exists():
            console.print(
                f"  {bullet} [bold]{agent_name}[/bold]{default_label}",
                style=COLORS["primary"],
            )
            console.print(f"    {agent_path}", style=COLORS["dim"])
        else:
            console.print(
                f"  {bullet} [bold]{agent_name}[/bold]{default_label}"
                " [dim](incomplete)[/dim]",
                style=COLORS["tool"],
            )
            console.print(f"    {agent_path}", style=COLORS["dim"])

    console.print()


def reset_agent(agent_name: str, source_agent: str | None = None) -> None:
    """Reset an agent to default or copy from another agent."""
    agent_dir = settings.get_agent_dir(agent_name)

    if source_agent:
        source_dir = settings.get_agent_dir(source_agent)
        source_md = source_dir / "AGENTS.md"

        if not source_md.exists():
            console.print(
                f"[bold red]Error:[/bold red] Source agent '{source_agent}' not found "
                "or has no AGENTS.md"
            )
            return

        source_content = source_md.read_text()
        action_desc = f"contents of agent '{source_agent}'"
    else:
        source_content = get_default_coding_instructions()
        action_desc = "default"

    if agent_dir.exists():
        shutil.rmtree(agent_dir)
        console.print(
            f"Removed existing agent directory: {agent_dir}", style=COLORS["tool"]
        )

    agent_dir.mkdir(parents=True, exist_ok=True)
    agent_md = agent_dir / "AGENTS.md"
    agent_md.write_text(source_content)

    console.print(
        f"{get_glyphs().checkmark} Agent '{agent_name}' reset to {action_desc}",
        style=COLORS["primary"],
    )
    console.print(f"Location: {agent_dir}\n", style=COLORS["dim"])


def get_system_prompt(assistant_id: str, sandbox_type: str | None = None) -> str:
    """Get the base system prompt for the agent.

    Loads the immutable system prompt from `system_prompt.md` and
    interpolates dynamic sections (model identity, working directory,
    skills path).

    Args:
        assistant_id: The agent identifier for path references
        sandbox_type: Type of sandbox provider
            (`'daytona'`, `'langsmith'`, `'modal'`, `'runloop'`).

            If `None`, agent is operating in local mode.

    Returns:
        The system prompt string

    Example:
        ```txt
        You are running as model {MODEL} (provider: {PROVIDER}).

        Your context window is {CONTEXT_WINDOW} tokens.

        ... {CONDITIONAL SECTIONS} ...
        ```
    """
    template = (Path(__file__).parent / "system_prompt.md").read_text()

    skills_path = f"~/.rlmagents/{assistant_id}/skills/"

    # Build model identity section
    model_identity_section = ""
    if settings.model_name:
        model_identity_section = (
            f"### Model Identity\n\nYou are running as model `{settings.model_name}`"
        )
        if settings.model_provider:
            model_identity_section += f" (provider: {settings.model_provider})"
        model_identity_section += ".\n"
        if settings.model_context_limit:
            model_identity_section += (
                f"Your context window is {settings.model_context_limit:,} tokens.\n"
            )
        model_identity_section += "\n"

    # Build working directory section (local vs sandbox)
    if sandbox_type:
        working_dir = get_default_working_dir(sandbox_type)
        working_dir_section = (
            f"### Current Working Directory\n\n"
            f"You are operating in a **remote Linux sandbox** at `{working_dir}`.\n\n"
            f"All code execution and file operations happen in this sandbox "
            f"environment.\n\n"
            f"**Important:**\n"
            f"- The CLI is running locally on the user's machine, but you execute "
            f"code remotely\n"
            f"- Use `{working_dir}` as your working directory for all operations\n\n"
        )
    else:
        cwd = Path.cwd()
        working_dir_section = (
            f"### Current Working Directory\n\n"
            f"The filesystem backend is currently operating in: `{cwd}`\n\n"
            f"### File System and Paths\n\n"
            f"**IMPORTANT - Path Handling:**\n"
            f"- All file paths must be absolute paths (e.g., `{cwd}/file.txt`)\n"
            f"- Use the working directory to construct absolute paths\n"
            f"- Example: To create a file in your working directory, "
            f"use `{cwd}/research_project/file.md`\n"
            f"- Never use relative paths - always construct full absolute paths\n\n"
        )

    return (
        template.replace("{model_identity_section}", model_identity_section)
        .replace("{working_dir_section}", working_dir_section)
        .replace("{skills_path}", skills_path)
    )


def _format_write_file_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """Format write_file tool call for approval prompt.

    Returns:
        Formatted description string for the write_file tool call.
    """
    args = tool_call["args"]
    file_path = args.get("file_path", "unknown")
    content = args.get("content", "")

    action = "Overwrite" if Path(file_path).exists() else "Create"
    line_count = len(content.splitlines())

    return f"File: {file_path}\nAction: {action} file\nLines: {line_count}"


def _format_edit_file_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """Format edit_file tool call for approval prompt.

    Returns:
        Formatted description string for the edit_file tool call.
    """
    args = tool_call["args"]
    file_path = args.get("file_path", "unknown")
    replace_all = bool(args.get("replace_all", False))

    scope = "all occurrences" if replace_all else "single occurrence"
    return f"File: {file_path}\nAction: Replace text ({scope})"


def _format_web_search_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """Format web_search tool call for approval prompt.

    Returns:
        Formatted description string for the web_search tool call.
    """
    args = tool_call["args"]
    query = args.get("query", "unknown")
    max_results = args.get("max_results", 5)

    return (
        f"Query: {query}\nMax results: {max_results}\n\n"
        f"{get_glyphs().warning}  This will use Tavily API credits"
    )


def _format_fetch_url_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """Format fetch_url tool call for approval prompt.

    Returns:
        Formatted description string for the fetch_url tool call.
    """
    args = tool_call["args"]
    url = args.get("url", "unknown")
    timeout = args.get("timeout", 30)

    return (
        f"URL: {url}\nTimeout: {timeout}s\n\n"
        f"{get_glyphs().warning}  Will fetch and convert web content to markdown"
    )


def _format_task_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """Format task (subagent) tool call for approval prompt.

    The task tool signature is: task(description: str, subagent_type: str)
    The description contains all instructions that will be sent to the subagent.

    Returns:
        Formatted description string for the task tool call.
    """
    args = tool_call["args"]
    description = args.get("description", "unknown")
    subagent_type = args.get("subagent_type", "unknown")

    # Truncate description if too long for display
    description_preview = description
    if len(description) > 500:
        description_preview = description[:500] + "..."

    glyphs = get_glyphs()
    separator = glyphs.box_horizontal * 40
    warning_msg = "Subagent will have access to file operations and shell commands"
    return (
        f"Subagent Type: {subagent_type}\n\n"
        f"Task Instructions:\n"
        f"{separator}\n"
        f"{description_preview}\n"
        f"{separator}\n\n"
        f"{glyphs.warning}  {warning_msg}"
    )


def _format_execute_description(
    tool_call: ToolCall, _state: AgentState[Any], _runtime: Runtime[Any]
) -> str:
    """Format execute tool call for approval prompt.

    Returns:
        Formatted description string for the execute tool call.
    """
    args = tool_call["args"]
    command = args.get("command", "N/A")
    return f"Execute Command: {command}\nWorking Directory: {Path.cwd()}"


def _add_interrupt_on() -> dict[str, InterruptOnConfig]:
    """Configure human-in-the-loop interrupt settings for all gated tools.

    Every tool that can have side effects or access external resources
    (shell execution, file writes/edits, web search, URL fetch, task
    delegation) is gated behind an approval prompt unless auto-approve
    is enabled.

    Returns:
        Dictionary mapping tool names to their interrupt configuration.
    """
    execute_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_execute_description,  # type: ignore[typeddict-item]
    }

    write_file_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_write_file_description,  # type: ignore[typeddict-item]
    }

    edit_file_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_edit_file_description,  # type: ignore[typeddict-item]
    }

    web_search_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_web_search_description,  # type: ignore[typeddict-item]
    }

    fetch_url_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_fetch_url_description,  # type: ignore[typeddict-item]
    }

    task_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_task_description,  # type: ignore[typeddict-item]
    }

    return {
        "execute": execute_interrupt_config,
        "write_file": write_file_interrupt_config,
        "edit_file": edit_file_interrupt_config,
        "web_search": web_search_interrupt_config,
        "fetch_url": fetch_url_interrupt_config,
        "task": task_interrupt_config,
    }


def create_cli_agent(
    model: str | BaseChatModel,
    assistant_id: str,
    *,
    harness: HarnessType = "rlmagents",
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    sandbox: SandboxBackendProtocol | None = None,
    sandbox_type: str | None = None,
    system_prompt: str | None = None,
    auto_approve: bool = False,
    enable_memory: bool = True,
    enable_skills: bool = True,
    enable_shell: bool = True,
    checkpointer: BaseCheckpointSaver | None = None,
) -> tuple[Pregel, CompositeBackend]:
    """Create a CLI-configured agent with flexible options.

    This is the main entry point for creating an rlmagents CLI agent, usable
    both internally and from external code (e.g., benchmarking frameworks).

    Args:
        model: LLM model to use (e.g., `'anthropic:claude-sonnet-4-5-20250929'`)
        assistant_id: Agent identifier for memory/state storage
        harness: Agent harness runtime to use.

            - `'rlmagents'`: Use the rlmagents recursive harness (default)
            - `'deepagents'`: Compatibility alias for `'rlmagents'`
        tools: Additional tools to provide to agent
        sandbox: Optional sandbox backend for remote execution
            (e.g., `ModalBackend`).

            If `None`, uses local filesystem + shell.
        sandbox_type: Type of sandbox provider
            (`'daytona'`, `'langsmith'`, `'modal'`, `'runloop'`).
            Used for system prompt generation.
        system_prompt: Override the default system prompt.

            If `None`, generates one based on `sandbox_type` and `assistant_id`.
        auto_approve: If `True`, no tools trigger human-in-the-loop
            interrupts — all calls (shell execution, file writes/edits,
            web search, URL fetch) run automatically.

            If `False`, tools pause for user confirmation via the approval menu.
            See `_add_interrupt_on` for the full list of gated tools.
        enable_memory: Enable `MemoryMiddleware` for persistent memory
        enable_skills: Enable `SkillsMiddleware` for custom agent skills
        enable_shell: Enable shell execution via `CLIShellBackend`
            (only in local mode). When enabled, the `execute` tool is available.
        checkpointer: Optional checkpointer for session persistence.

            If `None`, uses `InMemorySaver` (no persistence across
            CLI invocations).

    Returns:
        2-tuple of `(agent_graph, backend)`

            - `agent_graph`: Configured LangGraph Pregel instance ready
                for execution
            - `composite_backend`: `CompositeBackend` for file operations

    Raises:
        ValueError: If `harness` is not a supported value.
    """
    tools = tools or []

    # Setup agent directory for persistent memory (if enabled)
    if enable_memory or enable_skills:
        agent_dir = settings.ensure_agent_dir(assistant_id)
        agent_md = agent_dir / "AGENTS.md"
        if not agent_md.exists():
            # Create empty file for user customizations
            # Base instructions are loaded fresh from get_system_prompt()
            agent_md.touch()

    # Skills directories (if enabled)
    if enable_skills:
        settings.ensure_user_skills_dir(assistant_id)

    # Load custom subagents from filesystem
    custom_subagents_map: dict[str, SubAgent | CompiledSubAgent] = {}
    user_agent_dirs = settings.get_user_agents_dirs(assistant_id)
    project_agent_dirs = settings.get_project_agents_dirs()

    for user_dir in user_agent_dirs:
        for subagent_meta in list_subagents(
            user_agents_dir=user_dir,
            project_agents_dir=None,
        ):
            subagent: SubAgent = {
                "name": subagent_meta["name"],
                "description": subagent_meta["description"],
                "system_prompt": subagent_meta["system_prompt"],
            }
            if subagent_meta["model"]:
                subagent["model"] = subagent_meta["model"]
            custom_subagents_map[subagent_meta["name"]] = subagent

    for project_dir in project_agent_dirs:
        for subagent_meta in list_subagents(
            user_agents_dir=None,
            project_agents_dir=project_dir,
        ):
            subagent: SubAgent = {
                "name": subagent_meta["name"],
                "description": subagent_meta["description"],
                "system_prompt": subagent_meta["system_prompt"],
            }
            if subagent_meta["model"]:
                subagent["model"] = subagent_meta["model"]
            custom_subagents_map[subagent_meta["name"]] = subagent

    custom_subagents = list(custom_subagents_map.values())

    # Build middleware stack based on enabled features
    agent_middleware = []

    def _dedupe_paths(paths: list[Path]) -> list[str]:
        """Convert paths to unique strings while preserving order.

        Returns:
            Ordered unique path strings.
        """
        seen: set[str] = set()
        ordered: list[str] = []
        for path in paths:
            path_str = str(path)
            if path_str in seen:
                continue
            seen.add(path_str)
            ordered.append(path_str)
        return ordered

    # Add memory middleware
    if enable_memory:
        memory_source_paths = settings.get_user_agent_md_paths(assistant_id)
        memory_source_paths.extend(settings.get_project_agent_md_paths())
        memory_sources = _dedupe_paths(memory_source_paths)

        agent_middleware.append(
            MemoryMiddleware(
                backend=FilesystemBackend(),
                sources=memory_sources,
            )
        )

    # Add skills middleware
    if enable_skills:
        # Built-in first (lowest precedence), then user, then project (highest)
        source_paths = [settings.get_built_in_skills_dir()]
        source_paths.extend(settings.get_user_skills_dirs(assistant_id))
        source_paths.extend(settings.get_project_skills_dirs())
        sources = _dedupe_paths(source_paths)

        agent_middleware.append(
            SkillsMiddleware(
                backend=FilesystemBackend(),
                sources=sources,
            )
        )

    # CONDITIONAL SETUP: Local vs Remote Sandbox
    if sandbox is None:
        # ========== LOCAL MODE ==========
        if enable_shell:
            # Create environment for shell commands
            # Restore user's original LANGSMITH_PROJECT so their code traces separately
            shell_env = os.environ.copy()
            if settings.user_langchain_project:
                shell_env["LANGSMITH_PROJECT"] = settings.user_langchain_project

            # Use CLIShellBackend for filesystem + shell execution.
            # Provides `execute` tool via FilesystemMiddleware with per-command
            # timeout support.
            backend = CLIShellBackend(
                root_dir=Path.cwd(),
                inherit_env=True,
                env=shell_env,
            )
        else:
            # No shell access - use plain FilesystemBackend
            backend = FilesystemBackend()

        # Local context middleware (git info, directory tree, etc.)
        agent_middleware.append(LocalContextMiddleware())
    else:
        # ========== REMOTE SANDBOX MODE ==========
        backend = sandbox  # Remote sandbox (ModalBackend, etc.)
        # Note: Shell middleware not used in sandbox mode
        # File operations and execute tool are provided by the sandbox backend

    # Get or use custom system prompt
    if system_prompt is None:
        system_prompt = get_system_prompt(
            assistant_id=assistant_id, sandbox_type=sandbox_type
        )

    # Configure interrupt_on based on auto_approve setting
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None
    if auto_approve:
        # No interrupts - all tools run automatically
        interrupt_on = {}
    else:
        # Full HITL for destructive operations
        interrupt_on = _add_interrupt_on()  # type: ignore[assignment]

    # Set up composite backend with routing
    # For local FilesystemBackend, route large tool results to /tmp to avoid polluting
    # the working directory. For sandbox backends, no special routing is needed.
    if sandbox is None:
        # Local mode: Route large results to a unique temp directory
        large_results_backend = FilesystemBackend(
            root_dir=tempfile.mkdtemp(prefix="rlmagents_large_results_"),
            virtual_mode=True,
        )
        conversation_history_backend = FilesystemBackend(
            root_dir=tempfile.mkdtemp(prefix="rlmagents_conversation_history_"),
            virtual_mode=True,
        )
        composite_backend = CompositeBackend(
            default=backend,
            routes={
                "/large_tool_results/": large_results_backend,
                "/conversation_history/": conversation_history_backend,
            },
        )
    else:
        # Sandbox mode: No special routing needed
        composite_backend = CompositeBackend(
            default=backend,
            routes={},
        )

    # Create the agent
    # Use provided checkpointer or fallback to InMemorySaver
    resolved_harness: HarnessType = harness
    if (
        resolved_harness in {"rlmagents", "deepagents"}
        and sandbox is None
        and enable_shell
    ):
        # Patch FilesystemMiddleware so the SDK constructs our subclass with
        # per-command timeout support on the execute tool. Only needed in local
        # shell mode -- remote sandbox backends do not accept the timeout kwarg.
        patch_filesystem_middleware()
    final_checkpointer = checkpointer if checkpointer is not None else InMemorySaver()

    if resolved_harness in {"rlmagents", "deepagents"}:
        # Enable RLM recursive sub_query()/llm_query() by default in CLI mode.
        # If the provided model is an instantiated BaseChatModel, reuse it for
        # sub-queries so the full RLM toolchain is available.
        sub_query_model = model if isinstance(model, BaseChatModel) else None
        factory = (
            create_rlm_agent
            if resolved_harness == "rlmagents"
            else create_deep_agent
        )
        agent = factory(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            backend=composite_backend,
            middleware=agent_middleware,
            interrupt_on=interrupt_on,
            checkpointer=final_checkpointer,
            subagents=custom_subagents or None,
            sub_query_model=sub_query_model,
        ).with_config(config)
    else:
        msg = f"Unsupported harness: {harness!r}"
        raise ValueError(msg)
    return agent, composite_backend
