"""LangChain tool factories for RLM middleware."""

from __future__ import annotations

import asyncio
import bz2
import difflib
import gzip
import json
import lzma
import shutil
import subprocess
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from langchain_core.tools import BaseTool, StructuredTool

from rlmagents.session_manager import RLMSessionManager
from rlmagents.types import Evidence

RLMToolProfile = Literal["full", "reasoning", "core"]
DEFAULT_RLM_TOOL_PROFILE: RLMToolProfile = "full"
RLM_TOOL_NAMES = frozenset(
    {
        "load_context",
        "load_file_context",
        "list_contexts",
        "diff_contexts",
        "save_session",
        "load_session",
        "peek_context",
        "search_context",
        "semantic_search",
        "chunk_context",
        "cross_context_search",
        "rg_search",
        "exec_python",
        "get_variable",
        "think",
        "evaluate_progress",
        "summarize_so_far",
        "get_evidence",
        "finalize",
        "get_status",
        "rlm_tasks",
        "validate_recipe",
        "estimate_recipe",
        "run_recipe",
        "run_recipe_code",
        "configure_rlm",
    }
)

_RLM_TOOL_PROFILES: dict[RLMToolProfile, tuple[str, ...]] = {
    "full": (
        "load_context",
        "load_file_context",
        "list_contexts",
        "diff_contexts",
        "save_session",
        "load_session",
        "peek_context",
        "search_context",
        "semantic_search",
        "chunk_context",
        "cross_context_search",
        "rg_search",
        "exec_python",
        "get_variable",
        "think",
        "evaluate_progress",
        "summarize_so_far",
        "get_evidence",
        "finalize",
        "get_status",
        "rlm_tasks",
        "validate_recipe",
        "estimate_recipe",
        "run_recipe",
        "run_recipe_code",
        "configure_rlm",
    ),
    "reasoning": (
        "load_context",
        "load_file_context",
        "list_contexts",
        "save_session",
        "load_session",
        "peek_context",
        "search_context",
        "semantic_search",
        "chunk_context",
        "cross_context_search",
        "rg_search",
        "exec_python",
        "get_variable",
        "think",
        "evaluate_progress",
        "summarize_so_far",
        "get_evidence",
        "finalize",
        "get_status",
        "rlm_tasks",
    ),
    "core": (
        "load_context",
        "load_file_context",
        "list_contexts",
        "peek_context",
        "search_context",
        "semantic_search",
        "chunk_context",
        "exec_python",
        "think",
        "evaluate_progress",
        "summarize_so_far",
        "get_evidence",
        "finalize",
        "get_status",
    ),
}


def _validate_tool_names(names: Sequence[str], *, source: str) -> None:
    unknown = sorted({name for name in names if name not in RLM_TOOL_NAMES})
    if unknown:
        known_names = ", ".join(sorted(RLM_TOOL_NAMES))
        unknown_names = ", ".join(unknown)
        raise ValueError(
            f"Unknown tools in {source}: {unknown_names}. "
            f"Valid tools: {known_names}"
        )


def _get_tool_factories() -> dict[str, Callable[[RLMSessionManager], BaseTool]]:
    return {
        "load_context": _create_load_context_tool,
        "load_file_context": _create_load_file_context_tool,
        "list_contexts": _create_list_contexts_tool,
        "diff_contexts": _create_diff_contexts_tool,
        "save_session": _create_save_session_tool,
        "load_session": _create_load_session_tool,
        "peek_context": _create_peek_context_tool,
        "search_context": _create_search_context_tool,
        "semantic_search": _create_semantic_search_tool,
        "chunk_context": _create_chunk_context_tool,
        "cross_context_search": _create_cross_context_search_tool,
        "rg_search": _create_rg_search_tool,
        "exec_python": _create_exec_python_tool,
        "get_variable": _create_get_variable_tool,
        "think": _create_think_tool,
        "evaluate_progress": _create_evaluate_progress_tool,
        "summarize_so_far": _create_summarize_so_far_tool,
        "get_evidence": _create_get_evidence_tool,
        "finalize": _create_finalize_tool,
        "get_status": _create_get_status_tool,
        "rlm_tasks": _create_rlm_tasks_tool,
        "validate_recipe": _create_validate_recipe_tool,
        "estimate_recipe": _create_estimate_recipe_tool,
        "run_recipe": _create_run_recipe_tool,
        "run_recipe_code": _create_run_recipe_code_tool,
        "configure_rlm": _create_configure_rlm_tool,
    }


def _build_rlm_tools(
    manager: RLMSessionManager,
    *,
    profile: RLMToolProfile = DEFAULT_RLM_TOOL_PROFILE,
    include_tools: Sequence[str] = (),
    exclude_tools: Sequence[str] = (),
) -> list[BaseTool]:
    """Build RLM tools for a profile with optional include/exclude overrides."""
    _validate_tool_names(include_tools, source="include_tools")
    _validate_tool_names(exclude_tools, source="exclude_tools")
    if profile not in _RLM_TOOL_PROFILES:
        valid_profiles = ", ".join(sorted(_RLM_TOOL_PROFILES))
        raise ValueError(f"Unknown RLM tool profile '{profile}'. Valid profiles: {valid_profiles}")

    selected: list[str] = list(_RLM_TOOL_PROFILES[profile])
    for name in include_tools:
        if name not in selected:
            selected.append(name)

    excluded = set(exclude_tools)
    factories = _get_tool_factories()
    return [factories[name](manager) for name in selected if name not in excluded]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_meta(meta: Any) -> str:
    """Format ContextMetadata to a readable string."""
    return (
        f"Format: {meta.format.value}, "
        f"{meta.size_chars:,} chars, "
        f"{meta.size_lines:,} lines, "
        f"~{meta.size_tokens_estimate:,} tokens"
    )


def _truncate(text: str, max_chars: int = 50_000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated at {max_chars:,} chars]"


def _split_csv_values(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _decode_text_bytes(raw: bytes, *, encoding: str, path: Path) -> str:
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError as exc:
        raise UnicodeDecodeError(
            exc.encoding,
            exc.object,
            exc.start,
            exc.end,
            (
                f"{exc.reason}. Failed to decode '{path}' using encoding "
                f"'{encoding}'."
            ),
        ) from exc


def _read_text_from_path(path: Path, *, encoding: str) -> tuple[str, str | None]:
    suffix = path.suffix.lower()
    if suffix == ".gz":
        raw = gzip.decompress(path.read_bytes())
        return _decode_text_bytes(raw, encoding=encoding, path=path), "gzip"
    if suffix == ".bz2":
        raw = bz2.decompress(path.read_bytes())
        return _decode_text_bytes(raw, encoding=encoding, path=path), "bz2"
    if suffix in {".xz", ".lzma"}:
        raw = lzma.decompress(path.read_bytes())
        return _decode_text_bytes(raw, encoding=encoding, path=path), "lzma"
    return path.read_text(encoding=encoding), None


def _format_pressure_status(status: dict[str, object]) -> str:
    """Format compacted-state metadata into a short status snippet."""
    if not status:
        return "not initialized"
    if not status.get("compacted"):
        return "not compacted"
    reason = status.get("last_compaction_reason")
    count = status.get("compaction_count")
    if reason is None or count is None:
        return "compacted"
    return f"compacted ({reason}, count={count})"


def _evidence_summary_lines(
    evidence: list[Evidence],
    *,
    max_items: int = 4,
) -> list[str]:
    """Build a compact evidence summary for user-facing reports."""
    if not evidence:
        return ["No evidence captured yet."]

    counts: dict[str, int] = {}
    for item in evidence:
        counts[item.source] = counts.get(item.source, 0) + 1

    parts = ["Evidence summary:"]
    for source, count in sorted(counts.items()):
        parts.append(f"- {source}: {count}")

    parts.append("Recent evidence:")
    for idx, item in enumerate(evidence[-max_items:], 1):
        header = item.source
        if item.source_op:
            header += f" ({item.source_op})"
        if item.command_exit_status is not None:
            header += f" [exit {item.command_exit_status}]"
        if item.file_path:
            header += f" file={item.file_path}"
        parts.append(f"{idx}. {header}: {item.snippet[:120]}")
    return parts


def _render_final_report(
    manager: RLMSessionManager,
    *,
    context_id: str,
    answer: str,
    confidence: str = "medium",
    reasoning_summary: str = "",
    completion_source: str = "finalize",
) -> str:
    """Render a final answer block shared by finalize and sentinel completion."""
    session = manager.get_session(context_id)
    parts = [f"## Final Answer\n\n{answer}"]
    parts.append(f"\n**Confidence:** {confidence}")
    parts.append(f"**Completion source:** {completion_source}")

    if reasoning_summary:
        parts.append(f"**Reasoning:** {reasoning_summary}")

    if session:
        parts.append(
            f"\n**Analysis stats:** {session.iterations} iterations, "
            f"{len(session.evidence)} evidence items, "
            f"{len(session.think_history)} think steps, "
            f"{len(session.hist)} hist entries"
        )
        if manager.get_context_pressure_status(context_id).get("compacted"):
            compacted = manager.get_context_pressure_status(context_id)
            parts.append(
                f"**Context pressure:** compacted={compacted.get('compaction_count', 0)} "
                f"(reason: {compacted.get('last_compaction_reason') or 'n/a'})"
            )
        parts.extend(_evidence_summary_lines(session.evidence))

    return "\n".join(parts)


def _record_evidence(
    manager: RLMSessionManager,
    session_id: str,
    source: str,
    source_op: str,
    snippet: str,
    *,
    line_range: tuple[int, int] | None = None,
    pattern: str | None = None,
    file_path: str | None = None,
    command_exit_status: int | None = None,
) -> None:
    """Record evidence with extended metadata and consistent provenance fields."""
    session = manager.get_session(session_id)
    if session is None:
        return
    session.add_evidence(
        Evidence(
            source=source,  # type: ignore[arg-type]
            source_op=source_op,
            context_id=session_id,
            file_path=file_path,
            line_range=line_range,
            pattern=pattern,
            snippet=snippet,
            command_exit_status=command_exit_status,
        )
    )


# ===========================================================================
# Context Tools
# ===========================================================================


def _create_load_context_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_load_context(
        content: Annotated[str, "Text content to load into the isolated context"],
        context_id: Annotated[str, "Session identifier"] = "default",
        format_hint: Annotated[str, "Format hint: auto, text, json, csv, code"] = "auto",
        line_number_base: Annotated[int, "Line numbering base: 0 or 1"] = 1,
    ) -> str:
        meta = manager.create_session(
            content,
            context_id=context_id,
            format_hint=format_hint,
            line_number_base=line_number_base,
        )
        return f"Context '{context_id}' loaded. {_fmt_meta(meta)}"

    async def async_load_context(
        content: Annotated[str, "Text content to load into the isolated context"],
        context_id: Annotated[str, "Session identifier"] = "default",
        format_hint: Annotated[str, "Format hint: auto, text, json, csv, code"] = "auto",
        line_number_base: Annotated[int, "Line numbering base: 0 or 1"] = 1,
    ) -> str:
        return sync_load_context(content, context_id, format_hint, line_number_base)

    return StructuredTool.from_function(
        name="load_context",
        description=(
            "Load text content into an isolated RLM context for analysis. "
            "Use search_context/peek_context/exec_python to explore it. "
            "Supports multiple contexts via context_id."
        ),
        func=sync_load_context,
        coroutine=async_load_context,
    )


def _create_load_file_context_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_load_file_context(
        path: Annotated[str, "Filesystem path to load into an isolated context"],
        context_id: Annotated[str, "Session identifier"] = "default",
        format_hint: Annotated[str, "Format hint: auto, text, json, csv, code"] = "auto",
        line_number_base: Annotated[int, "Line numbering base: 0 or 1"] = 1,
        encoding: Annotated[str, "Text encoding for file reads"] = "utf-8",
        max_chars: Annotated[
            int,
            "Optional character cap (0 keeps full file)",
        ] = 0,
    ) -> str:
        file_path = Path(path).expanduser()
        try:
            resolved_path = file_path.resolve(strict=True)
        except FileNotFoundError:
            return f"File not found: {file_path}"
        except OSError as exc:
            return f"Unable to resolve path '{file_path}': {exc}"

        if not resolved_path.is_file():
            return f"Path is not a file: {resolved_path}"

        try:
            content, compression = _read_text_from_path(resolved_path, encoding=encoding)
        except UnicodeDecodeError as exc:
            return (
                f"Failed to decode '{resolved_path}' using encoding '{encoding}': {exc}. "
                "Use a different encoding or convert the file to UTF-8 first."
            )
        except OSError as exc:
            return f"Failed to read '{resolved_path}': {exc}"
        except Exception as exc:
            return f"Failed to load '{resolved_path}': {exc}"

        truncated = False
        if max_chars > 0 and len(content) > max_chars:
            content = content[:max_chars]
            truncated = True

        meta = manager.create_session(
            content,
            context_id=context_id,
            format_hint=format_hint,
            line_number_base=line_number_base,
        )
        snippet = (
            f"Loaded file '{resolved_path}' into context '{context_id}' "
            f"({meta.size_chars:,} chars)."
        )
        _record_evidence(
            manager=manager,
            session_id=context_id,
            source="action",
            source_op="load_file_context",
            snippet=snippet,
            file_path=str(resolved_path),
        )

        msg = f"File '{resolved_path}' loaded into context '{context_id}'. {_fmt_meta(meta)}"
        if compression:
            msg += f" [decompressed:{compression}]"
        if truncated:
            msg += f" [truncated to {max_chars:,} chars]"
        return msg

    async def async_load_file_context(
        path: Annotated[str, "Filesystem path to load into an isolated context"],
        context_id: Annotated[str, "Session identifier"] = "default",
        format_hint: Annotated[str, "Format hint: auto, text, json, csv, code"] = "auto",
        line_number_base: Annotated[int, "Line numbering base: 0 or 1"] = 1,
        encoding: Annotated[str, "Text encoding for file reads"] = "utf-8",
        max_chars: Annotated[
            int,
            "Optional character cap (0 keeps full file)",
        ] = 0,
    ) -> str:
        return sync_load_file_context(
            path,
            context_id,
            format_hint,
            line_number_base,
            encoding,
            max_chars,
        )

    return StructuredTool.from_function(
        name="load_file_context",
        description=(
            "Load a file directly into an isolated RLM context without copying file "
            "contents into the chat transcript."
        ),
        func=sync_load_file_context,
        coroutine=async_load_file_context,
    )


def _create_list_contexts_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_list_contexts() -> str:
        sessions = manager.list_sessions()
        if not sessions:
            return "No active contexts."
        lines = []
        for cid, info in sessions.items():
            lines.append(
                f"- {cid}: {info['format']}, {info['size_chars']:,} chars, "
                f"{info['iterations']} iterations, {info['hist_count']} hist, "
                f"{info['evidence_count']} evidence"
            )
        return "\n".join(lines)

    async def async_list_contexts() -> str:
        return sync_list_contexts()

    return StructuredTool.from_function(
        name="list_contexts",
        description="List all active RLM context sessions with metadata.",
        func=sync_list_contexts,
        coroutine=async_list_contexts,
    )


def _create_diff_contexts_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_diff_contexts(
        a: Annotated[str, "First context ID"],
        b: Annotated[str, "Second context ID"],
        context_lines: Annotated[int, "Lines of context around changes"] = 3,
    ) -> str:
        session_a = manager.get_session(a)
        session_b = manager.get_session(b)
        if session_a is None:
            return f"Context '{a}' not found."
        if session_b is None:
            return f"Context '{b}' not found."

        text_a = str(session_a.repl.get_variable("ctx") or "")
        text_b = str(session_b.repl.get_variable("ctx") or "")

        diff = difflib.unified_diff(
            text_a.splitlines(keepends=True),
            text_b.splitlines(keepends=True),
            fromfile=a,
            tofile=b,
            n=context_lines,
        )
        result = "".join(diff)
        if not result:
            return "No differences found."
        return _truncate(result, 30_000)

    async def async_diff_contexts(
        a: Annotated[str, "First context ID"],
        b: Annotated[str, "Second context ID"],
        context_lines: Annotated[int, "Lines of context around changes"] = 3,
    ) -> str:
        return sync_diff_contexts(a, b, context_lines)

    return StructuredTool.from_function(
        name="diff_contexts",
        description="Compare two RLM contexts using unified diff.",
        func=sync_diff_contexts,
        coroutine=async_diff_contexts,
    )


def _create_save_session_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_save_session(
        context_id: Annotated[str, "Session to save"] = "default",
        path: Annotated[str, "File path to write JSON (optional)"] = "",
    ) -> str:
        try:
            payload = manager.save_session(
                context_id=context_id,
                path=path if path else None,
            )
        except KeyError:
            return f"No session with context_id='{context_id}'."
        size = len(json.dumps(payload))
        msg = f"Session '{context_id}' serialized ({size:,} bytes)."
        if path:
            msg += f" Written to {path}."
        return msg

    async def async_save_session(
        context_id: Annotated[str, "Session to save"] = "default",
        path: Annotated[str, "File path to write JSON (optional)"] = "",
    ) -> str:
        return sync_save_session(context_id, path)

    return StructuredTool.from_function(
        name="save_session",
        description="Serialize an RLM session to JSON (memory pack). Optionally write to file.",
        func=sync_save_session,
        coroutine=async_save_session,
    )


def _create_load_session_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_load_session(
        path: Annotated[str, "File path to load JSON session from"],
        context_id: Annotated[str, "Override context ID (optional)"] = "",
    ) -> str:
        try:
            meta = manager.load_session_from_file(
                path=path,
                context_id=context_id if context_id else None,
            )
        except Exception as exc:
            return f"Failed to load session: {exc}"
        return f"Session loaded from {path}. {_fmt_meta(meta)}"

    async def async_load_session(
        path: Annotated[str, "File path to load JSON session from"],
        context_id: Annotated[str, "Override context ID (optional)"] = "",
    ) -> str:
        return sync_load_session(path, context_id)

    return StructuredTool.from_function(
        name="load_session",
        description="Load a previously saved RLM session from a JSON file.",
        func=sync_load_session,
        coroutine=async_load_session,
    )


# ===========================================================================
# Query Tools
# ===========================================================================


def _create_peek_context_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_peek_context(
        context_id: Annotated[str, "Session identifier"] = "default",
        start: Annotated[int, "Start position (chars or lines)"] = 0,
        end: Annotated[int, "End position (chars or lines, 0 = all)"] = 0,
        unit: Annotated[str, "Unit: 'chars' or 'lines'"] = "chars",
    ) -> str:
        session = manager.get_or_create_session(context_id)
        compact_status = manager._note_tool_activity(context_id, source_op="peek_context")

        ctx_text = str(session.repl.get_variable("ctx") or "")
        if not ctx_text:
            return f"Context '{context_id}' is empty."

        if unit == "lines":
            all_lines = ctx_text.splitlines()
            effective_end = end if end > 0 else len(all_lines)
            selected = all_lines[start:effective_end]
            result = "\n".join(
                f"{i + start + session.line_number_base}: {line}" for i, line in enumerate(selected)
            )
        else:
            effective_end = end if end > 0 else len(ctx_text)
            result = ctx_text[start:effective_end]

        snippet = result[:200]
        _record_evidence(
            manager=manager,
            session_id=context_id,
            source="peek",
            source_op="peek_context",
            snippet=snippet,
            line_range=(start, end) if unit == "lines" else None,
            pattern=None,
            command_exit_status=None,
        )

        status = f"\n[context_pressure:{_format_pressure_status(manager.get_context_pressure_status(context_id))}]"
        if compact_status:
            status += f" [auto-compacted due to pressure: {compact_status}]"

        return _truncate(result) + status

    async def async_peek_context(
        context_id: Annotated[str, "Session identifier"] = "default",
        start: Annotated[int, "Start position (chars or lines)"] = 0,
        end: Annotated[int, "End position (chars or lines, 0 = all)"] = 0,
        unit: Annotated[str, "Unit: 'chars' or 'lines'"] = "chars",
    ) -> str:
        return sync_peek_context(context_id, start, end, unit)

    return StructuredTool.from_function(
        name="peek_context",
        description=(
            "View a portion of the RLM context by character range or line range. "
            "Use unit='lines' for line-based slicing."
        ),
        func=sync_peek_context,
        coroutine=async_peek_context,
    )


def _create_search_context_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_search_context(
        pattern: Annotated[str, "Regex pattern to search for"],
        context_id: Annotated[str, "Session identifier"] = "default",
        context_lines: Annotated[int, "Lines of context around each match"] = 2,
        max_results: Annotated[int, "Maximum matches to return"] = 10,
    ) -> str:
        session = manager.get_or_create_session(context_id)
        compact_status = manager._note_tool_activity(context_id, source_op="search_context")

        search_fn = session.repl.get_helper("search")
        if search_fn is None:
            return "search helper not available"
        results = search_fn(pattern, context_lines=context_lines, max_results=max_results)

        if not results:
            return f"No matches for pattern: {pattern}"

        lines = []
        for r in results:
            line_num = r.get("line_num", "?")
            match_text = r.get("match", "")
            ctx_text = r.get("context", "")
            lines.append(f"Line {line_num}: {match_text}")
            if ctx_text:
                lines.append(f"  {ctx_text}")

        snippet = "\n".join(lines)[:300]
        _record_evidence(
            manager=manager,
            session_id=context_id,
            source="search",
            source_op="search_context",
            snippet=snippet,
            pattern=pattern,
        )

        output = _truncate("\n".join(lines))
        status = f"\n[context_pressure:{_format_pressure_status(manager.get_context_pressure_status(context_id))}]"
        if compact_status:
            status += f" [auto-compacted due to pressure: {compact_status}]"
        return output + status

    async def async_search_context(
        pattern: Annotated[str, "Regex pattern to search for"],
        context_id: Annotated[str, "Session identifier"] = "default",
        context_lines: Annotated[int, "Lines of context around each match"] = 2,
        max_results: Annotated[int, "Maximum matches to return"] = 10,
    ) -> str:
        return sync_search_context(pattern, context_id, context_lines, max_results)

    return StructuredTool.from_function(
        name="search_context",
        description=(
            "Regex search over an isolated RLM context. Returns matches with "
            "line numbers and surrounding context. Evidence is recorded automatically."
        ),
        func=sync_search_context,
        coroutine=async_search_context,
    )


def _create_semantic_search_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_semantic_search(
        query: Annotated[str, "Natural language search query"],
        context_id: Annotated[str, "Session identifier"] = "default",
        chunk_size: Annotated[int, "Characters per chunk"] = 1000,
        overlap: Annotated[int, "Overlap between chunks"] = 100,
        top_k: Annotated[int, "Number of results to return"] = 5,
    ) -> str:
        session = manager.get_or_create_session(context_id)
        compact_status = manager._note_tool_activity(context_id, source_op="semantic_search")

        sem_fn = session.repl.get_helper("semantic_search")
        if sem_fn is None:
            return "semantic_search helper not available"

        results = sem_fn(
            query,
            chunk_size=chunk_size,
            overlap=overlap,
            top_k=top_k,
        )
        if not results:
            return f"No semantic matches for: {query}"

        lines = []
        for i, r in enumerate(results, 1):
            score = r.get("score", 0)
            text = str(r.get("text") or r.get("chunk") or r.get("preview") or "")[:500]
            start_char = r.get("start_char")
            end_char = r.get("end_char")
            span = ""
            if isinstance(start_char, int) and isinstance(end_char, int):
                span = f", chars={start_char}-{end_char}"
            lines.append(f"[{i}] score={score:.3f}{span}\n{text}")

        snippet = "\n".join(lines)[:300]
        _record_evidence(
            manager=manager,
            session_id=context_id,
            source="search",
            source_op="semantic_search",
            snippet=snippet,
            pattern=query,
        )
        output = _truncate("\n\n".join(lines))
        status = f"\n[context_pressure:{_format_pressure_status(manager.get_context_pressure_status(context_id))}]"
        if compact_status:
            status += f" [auto-compacted due to pressure: {compact_status}]"
        return output + status

    async def async_semantic_search(
        query: Annotated[str, "Natural language search query"],
        context_id: Annotated[str, "Session identifier"] = "default",
        chunk_size: Annotated[int, "Characters per chunk"] = 1000,
        overlap: Annotated[int, "Overlap between chunks"] = 100,
        top_k: Annotated[int, "Number of results to return"] = 5,
    ) -> str:
        return sync_semantic_search(query, context_id, chunk_size, overlap, top_k)

    return StructuredTool.from_function(
        name="semantic_search",
        description=(
            "Lightweight semantic search over RLM context using hashed embeddings. "
            "Good for finding conceptually related passages."
        ),
        func=sync_semantic_search,
        coroutine=async_semantic_search,
    )


def _create_chunk_context_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_chunk_context(
        context_id: Annotated[str, "Session identifier"] = "default",
        chunk_size: Annotated[int, "Characters per chunk"] = 5000,
        overlap: Annotated[int, "Overlap between chunks"] = 200,
        max_chunks: Annotated[int, "Maximum chunk previews to return"] = 20,
    ) -> str:
        if chunk_size <= 0:
            return "chunk_size must be > 0."
        if overlap < 0:
            return "overlap must be >= 0."
        if overlap >= chunk_size:
            return "overlap must be < chunk_size."
        if max_chunks <= 0:
            return "max_chunks must be > 0."

        session = manager.get_or_create_session(context_id)
        compact_status = manager._note_tool_activity(context_id, source_op="chunk_context")
        chunk_fn = session.repl.get_helper("chunk")
        if chunk_fn is None:
            return "chunk helper not available"

        try:
            chunks = chunk_fn(chunk_size=chunk_size, overlap=overlap)
        except Exception as exc:
            return f"chunk_context failed: {exc}"

        if not chunks:
            return f"Context '{context_id}' is empty."

        ctx_text = str(session.repl.get_variable("ctx") or "")
        step = chunk_size - overlap
        metadata: list[dict[str, int]] = []
        if step > 0:
            for idx, chunk_text in enumerate(chunks):
                start_char = idx * step
                end_char = min(len(ctx_text), start_char + len(chunk_text))
                metadata.append(
                    {
                        "index": idx,
                        "start_char": start_char,
                        "end_char": end_char,
                        "size_chars": len(chunk_text),
                    }
                )
        else:
            cursor = 0
            for idx, chunk_text in enumerate(chunks):
                size = len(chunk_text)
                metadata.append(
                    {
                        "index": idx,
                        "start_char": cursor,
                        "end_char": cursor + size,
                        "size_chars": size,
                    }
                )
                cursor += size

        session.chunks = metadata

        shown = min(max_chunks, len(metadata))
        lines = [
            (
                f"Created {len(metadata)} chunks for context '{context_id}' "
                f"(chunk_size={chunk_size}, overlap={overlap})."
            )
        ]
        for item in metadata[:shown]:
            lines.append(
                f"- chunk[{item['index']}]: chars={item['start_char']}-{item['end_char']} "
                f"size={item['size_chars']}"
            )
        if len(metadata) > shown:
            lines.append(f"... {len(metadata) - shown} additional chunks omitted.")

        snippet = "\n".join(lines)[:300]
        _record_evidence(
            manager=manager,
            session_id=context_id,
            source="manual",
            source_op="chunk_context",
            snippet=snippet,
        )

        output = "\n".join(lines)
        output += (
            f"\n[context_pressure:{_format_pressure_status(manager.get_context_pressure_status(context_id))}]"
        )
        if compact_status:
            output += f" [auto-compacted due to pressure: {compact_status}]"
        return output

    async def async_chunk_context(
        context_id: Annotated[str, "Session identifier"] = "default",
        chunk_size: Annotated[int, "Characters per chunk"] = 5000,
        overlap: Annotated[int, "Overlap between chunks"] = 200,
        max_chunks: Annotated[int, "Maximum chunk previews to return"] = 20,
    ) -> str:
        return sync_chunk_context(context_id, chunk_size, overlap, max_chunks)

    return StructuredTool.from_function(
        name="chunk_context",
        description=(
            "Split a context into deterministic overlapping character chunks and "
            "store chunk metadata in-session for follow-up analysis."
        ),
        func=sync_chunk_context,
        coroutine=async_chunk_context,
    )


def _create_cross_context_search_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_cross_context_search(
        pattern: Annotated[str, "Regex pattern to search for across all contexts"],
        context_lines: Annotated[int, "Lines of context around each match"] = 2,
        max_results_per_context: Annotated[int, "Maximum matches to return per context"] = 5,
        contexts: Annotated[
            str | None, "Comma-separated context IDs to search (default: all)"
        ] = None,
    ) -> str:
        """Search across multiple RLM contexts simultaneously.

        This tool performs a regex search across all active contexts (or specified
        contexts) and returns results organized by context with source attribution.

        Args:
            pattern: Regex pattern to search for.
            context_lines: Lines of context around each match.
            max_results_per_context: Maximum matches to return per context.
            contexts: Optional comma-separated list of context IDs to search.
                     If None, searches all active contexts.

        Returns:
            Search results organized by context with source attribution.
        """
        sessions = manager.list_sessions()
        if not sessions:
            return "No active contexts to search."

        # Determine which contexts to search
        compact_status: dict[str, str | None] = {}
        if contexts:
            context_ids = [cid.strip() for cid in contexts.split(",") if cid.strip()]
        else:
            context_ids = list(sessions.keys())

        all_results = []

        for context_id in context_ids:
            session = manager.get_session(context_id)
            if session is None:
                all_results.append(f"Context '{context_id}' not found.")
                continue

            search_fn = session.repl.get_helper("search")
            if search_fn is None:
                all_results.append(f"Context '{context_id}': search helper not available")
                continue

            compact_marker = manager._note_tool_activity(
                context_id, source_op="cross_context_search"
            )

            results = search_fn(
                pattern, context_lines=context_lines, max_results=max_results_per_context
            )

            if results:
                compact_status[context_id] = compact_marker or _format_pressure_status(
                    manager.get_context_pressure_status(context_id)
                )
                ctx_results = [f"## Context: {context_id}"]
                for r in results:
                    line_num = r.get("line_num", "?")
                    match_text = r.get("match", "")
                    ctx_text = r.get("context", "")
                    ctx_results.append(f"### Line {line_num}: {match_text}")
                    if ctx_text:
                        ctx_results.append(f"```\n{ctx_text}\n```")

                all_results.append("\n".join(ctx_results))

                # Record evidence for this search
                snippet = "\n".join([r.get("match", "") for r in results[:3]])[:300]
                compact_status[context_id] = compact_status[context_id]
                _record_evidence(
                    manager=manager,
                    session_id=context_id,
                    source="cross_context_search",
                    source_op="cross_context_search",
                    snippet=snippet,
                    pattern=pattern,
                )
            else:
                all_results.append(f"## Context: {context_id}\nNo matches found.")
                compact_status[context_id] = _format_pressure_status(
                    manager.get_context_pressure_status(context_id)
                )

        if not all_results:
            return "No contexts were searched."

        summary_lines = [_truncate("\n\n".join(all_results))]
        pressure_lines = [
            f"{cid}:{status}" for cid, status in compact_status.items() if status
        ]
        if pressure_lines:
            summary_lines.append("Pressure: " + ", ".join(pressure_lines))
        return "\n".join(summary_lines)

    async def async_cross_context_search(
        pattern: Annotated[str, "Regex pattern to search for across all contexts"],
        context_lines: Annotated[int, "Lines of context around each match"] = 2,
        max_results_per_context: Annotated[int, "Maximum matches to return per context"] = 5,
        contexts: Annotated[
            str | None, "Comma-separated context IDs to search (default: all)"
        ] = None,
    ) -> str:
        return sync_cross_context_search(pattern, context_lines, max_results_per_context, contexts)

    return StructuredTool.from_function(
        name="cross_context_search",
        description=(
            "Search across multiple RLM contexts simultaneously using regex. "
            "Returns results organized by context with source attribution. "
            "Use 'contexts' parameter to search specific contexts (comma-separated), "
            "or leave empty to search all active contexts."
        ),
        func=sync_cross_context_search,
        coroutine=async_cross_context_search,
    )


def _create_rg_search_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_rg_search(
        pattern: Annotated[str, "Regex pattern to search for with ripgrep"],
        paths: Annotated[str, "Comma-separated paths to search"] = ".",
        glob: Annotated[str, "Comma-separated glob filters"] = "",
        max_results: Annotated[int, "Maximum matching lines to keep"] = 1000,
        load_context_id: Annotated[
            str,
            "Optional context ID to load full search results into",
        ] = "",
        ignore_case: Annotated[bool, "Use case-insensitive matching"] = False,
        fixed_strings: Annotated[bool, "Treat pattern as a literal string"] = False,
    ) -> str:
        if not pattern.strip():
            return "Pattern must be non-empty."
        if max_results <= 0:
            return "max_results must be > 0."

        rg_binary = shutil.which("rg")
        if rg_binary is None:
            return "ripgrep ('rg') is not available on PATH."

        resolved_paths = _split_csv_values(paths) or ["."]
        cmd: list[str] = [
            rg_binary,
            "--line-number",
            "--no-heading",
            "--color=never",
            "--max-count",
            str(max_results),
        ]
        if ignore_case:
            cmd.append("--ignore-case")
        if fixed_strings:
            cmd.append("--fixed-strings")
        for item in _split_csv_values(glob):
            cmd.extend(["--glob", item])
        cmd.append(pattern)
        cmd.extend(resolved_paths)

        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if completed.returncode not in {0, 1}:
            error_msg = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
            return f"rg_search failed (exit {completed.returncode}): {error_msg}"

        if completed.returncode == 1 or not completed.stdout.strip():
            return f"No matches for pattern: {pattern}"

        output = completed.stdout.strip()
        line_count = output.count("\n") + 1
        snippet = output[:300]

        context_id = load_context_id.strip() if load_context_id else ""
        if context_id:
            meta = manager.create_session(output, context_id=context_id, format_hint="text")
            _record_evidence(
                manager=manager,
                session_id=context_id,
                source="action",
                source_op="rg_search",
                snippet=snippet,
                pattern=pattern,
                command_exit_status=completed.returncode,
            )
            preview = _truncate(output, max_chars=3000)
            return (
                f"rg_search found {line_count} matches. "
                f"Results loaded into context '{context_id}'. {_fmt_meta(meta)}\n\n"
                f"{preview}"
            )

        return (
            f"rg_search found {line_count} matches across {len(resolved_paths)} path(s).\n"
            f"{_truncate(output)}"
        )

    async def async_rg_search(
        pattern: Annotated[str, "Regex pattern to search for with ripgrep"],
        paths: Annotated[str, "Comma-separated paths to search"] = ".",
        glob: Annotated[str, "Comma-separated glob filters"] = "",
        max_results: Annotated[int, "Maximum matching lines to keep"] = 1000,
        load_context_id: Annotated[
            str,
            "Optional context ID to load full search results into",
        ] = "",
        ignore_case: Annotated[bool, "Use case-insensitive matching"] = False,
        fixed_strings: Annotated[bool, "Treat pattern as a literal string"] = False,
    ) -> str:
        return await asyncio.to_thread(
            sync_rg_search,
            pattern,
            paths,
            glob,
            max_results,
            load_context_id,
            ignore_case,
            fixed_strings,
        )

    return StructuredTool.from_function(
        name="rg_search",
        description=(
            "Run ripgrep over local paths and optionally load full hits into an "
            "RLM context for follow-up search/analysis."
        ),
        func=sync_rg_search,
        coroutine=async_rg_search,
    )


def _create_exec_python_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_exec_python(
        code: Annotated[str, "Python code to execute in the sandboxed REPL"],
        context_id: Annotated[str, "Session identifier"] = "default",
    ) -> str:
        session = manager.get_or_create_session(context_id)
        compact_status = manager._note_tool_activity(context_id, source_op="exec_python")

        result = session.repl.execute(code)
        final_value = session.repl.get_variable("Final")
        final_set = final_value is not None

        parts = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(f"STDERR: {result.stderr}")
        if result.return_value is not None:
            rv = repr(result.return_value)
            if len(rv) > 10_000:
                rv = rv[:10_000] + "... [truncated]"
            parts.append(f"=> {rv}")
        if result.error:
            parts.append(f"ERROR: {result.error}")
        if result.variables_updated:
            parts.append(f"Variables updated: {', '.join(result.variables_updated)}")

        output = "\n".join(parts) if parts else "(no output)"

        # Record evidence for any citations made during execution
        exec_exit_code = 0 if result.error is None else 1
        for citation in session.repl._citations:
            _record_evidence(
                manager=manager,
                session_id=context_id,
                source="exec",
                source_op="exec_python",
                snippet=str(citation.get("snippet", ""))[:300],
                line_range=citation.get("line_range"),
                pattern=None,
                command_exit_status=exec_exit_code,
            )
        session.repl._citations.clear()

        manager.append_hist_entry(
            context_id,
            {
                "timestamp": datetime.now().isoformat(),
                "kind": "exec_python",
                "source_op": "exec_python",
                "context_id": context_id,
                "iteration": session.iterations,
                "code_preview": code,
                "stdout_chars": len(result.stdout),
                "stderr_chars": len(result.stderr),
                "stdout_preview": result.stdout,
                "stderr_preview": result.stderr,
                "return_value_present": result.return_value is not None,
                "variables_updated_count": len(result.variables_updated),
                "variables_updated_preview": result.variables_updated[:10],
                "execution_time_ms": round(result.execution_time_ms, 2),
                "error": result.error,
                "truncated_output": result.truncated,
                "final_set": final_set,
                "final_preview": str(final_value) if final_set else "",
            },
        )

        if manager.enable_final_sentinel:
            sentinel_value = manager.consume_final_sentinel(context_id)
            if sentinel_value is not None:
                return _render_final_report(
                    manager,
                    context_id=context_id,
                    answer=sentinel_value,
                    confidence="medium",
                    reasoning_summary="Completed via REPL `Final` sentinel.",
                    completion_source="Final sentinel",
                )

        output = _truncate(output)
        output += (
            f"\n[context_pressure:{_format_pressure_status(manager.get_context_pressure_status(context_id))}]"
        )
        if compact_status:
            output += f" [auto-compacted due to pressure: {compact_status}]"
        return output

    async def async_exec_python(
        code: Annotated[str, "Python code to execute in the sandboxed REPL"],
        context_id: Annotated[str, "Session identifier"] = "default",
    ) -> str:
        # Execute in a worker thread so REPL helpers like sub_query() can
        # safely bridge async model calls back to the main event loop.
        return await asyncio.to_thread(sync_exec_python, code, context_id)

    return StructuredTool.from_function(
        name="exec_python",
        description=(
            "Execute Python code in the sandboxed REPL with 100+ built-in helpers. "
            "The context is available as `ctx`. Helpers include search(), peek(), "
            "extract_*(), cite(), sub_query(), set_final(), and more."
        ),
        func=sync_exec_python,
        coroutine=async_exec_python,
    )


def _create_get_variable_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_get_variable(
        name: Annotated[str, "Variable name to retrieve from the REPL namespace"],
        context_id: Annotated[str, "Session identifier"] = "default",
    ) -> str:
        session = manager.get_session(context_id)
        if session is None:
            return f"No session '{context_id}'."
        val = session.repl.get_variable(name)
        if val is None:
            return f"Variable '{name}' not found."
        result = repr(val)
        return _truncate(result, 20_000)

    async def async_get_variable(
        name: Annotated[str, "Variable name to retrieve from the REPL namespace"],
        context_id: Annotated[str, "Session identifier"] = "default",
    ) -> str:
        return sync_get_variable(name, context_id)

    return StructuredTool.from_function(
        name="get_variable",
        description="Retrieve a variable from the RLM REPL namespace.",
        func=sync_get_variable,
        coroutine=async_get_variable,
    )


# ===========================================================================
# Reasoning Tools (5)
# ===========================================================================


def _create_think_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_think(
        question: Annotated[str, "The reasoning sub-step or question to structure"],
        context_id: Annotated[str, "Session identifier"] = "default",
        context_slice: Annotated[str, "Optional context excerpt to reason about"] = "",
    ) -> str:
        session = manager.get_or_create_session(context_id)
        session.iterations += 1
        session.think_history.append(question)

        step_num = len(session.think_history)
        parts = [f"## Think Step {step_num}\n\n**Question:** {question}"]

        if context_slice:
            parts.append(f"\n**Context excerpt:**\n```\n{context_slice[:2000]}\n```")

        parts.append(
            "\n**Instructions:** Analyze this question using available evidence and context. "
            "Use search_context or exec_python to gather more data if needed."
        )

        return "\n".join(parts)

    async def async_think(
        question: Annotated[str, "The reasoning sub-step or question to structure"],
        context_id: Annotated[str, "Session identifier"] = "default",
        context_slice: Annotated[str, "Optional context excerpt to reason about"] = "",
    ) -> str:
        return sync_think(question, context_id, context_slice)

    return StructuredTool.from_function(
        name="think",
        description=(
            "Structure a reasoning sub-step. Records the question in think_history "
            "and returns a structured prompt for analysis."
        ),
        func=sync_think,
        coroutine=async_think,
    )


def _create_evaluate_progress_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_evaluate_progress(
        current_understanding: Annotated[str, "Summary of current understanding"],
        confidence_score: Annotated[float, "Confidence score 0.0-1.0"] = 0.5,
        remaining_questions: Annotated[str, "Comma-separated remaining questions"] = "",
        context_id: Annotated[str, "Session identifier"] = "default",
    ) -> str:
        session = manager.get_or_create_session(context_id)
        session.confidence_history.append(confidence_score)

        prev_evidence = session.information_gain[-1] if session.information_gain else 0
        current_evidence = len(session.evidence)
        gain = current_evidence - prev_evidence
        session.information_gain.append(current_evidence)

        parts = [
            "## Progress Evaluation",
            f"**Confidence:** {confidence_score:.0%}",
            f"**Information gain:** +{gain} evidence items (total: {current_evidence})",
            f"**Iterations:** {session.iterations}",
            f"**Think steps:** {len(session.think_history)}",
        ]

        if remaining_questions:
            parts.append(f"**Remaining questions:** {remaining_questions}")

        trend = ""
        if len(session.confidence_history) >= 2:
            delta = session.confidence_history[-1] - session.confidence_history[-2]
            trend = "improving" if delta > 0 else "declining" if delta < 0 else "stable"
            parts.append(f"**Confidence trend:** {trend} ({delta:+.0%})")

        if confidence_score >= 0.8:
            parts.append("\nConsider using `finalize` if you have sufficient evidence.")
        elif confidence_score < 0.3:
            parts.append(
                "\nConfidence is low. Consider more exploration with search_context or exec_python."
            )

        return "\n".join(parts)

    async def async_evaluate_progress(
        current_understanding: Annotated[str, "Summary of current understanding"],
        confidence_score: Annotated[float, "Confidence score 0.0-1.0"] = 0.5,
        remaining_questions: Annotated[str, "Comma-separated remaining questions"] = "",
        context_id: Annotated[str, "Session identifier"] = "default",
    ) -> str:
        return sync_evaluate_progress(
            current_understanding,
            confidence_score,
            remaining_questions,
            context_id,
        )

    return StructuredTool.from_function(
        name="evaluate_progress",
        description=(
            "Self-evaluate reasoning progress. Tracks confidence history and "
            "information gain to guide the analysis."
        ),
        func=sync_evaluate_progress,
        coroutine=async_evaluate_progress,
    )


def _create_summarize_so_far_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_summarize_so_far(
        context_id: Annotated[str, "Session identifier"] = "default",
        include_evidence: Annotated[bool, "Include evidence summary"] = True,
        include_variables: Annotated[bool, "Include REPL variable names"] = True,
        clear_history: Annotated[bool, "Clear think_history after summarizing"] = False,
    ) -> str:
        session = manager.get_session(context_id)
        if session is None:
            return f"No session '{context_id}'."

        parts = [f"## Session Summary: {context_id}"]
        parts.append(f"**Iterations:** {session.iterations}")
        parts.append(f"**Context:** {_fmt_meta(session.meta)}")

        if session.think_history:
            recent_thinks = session.think_history[-10:]
            parts.append(f"\n### Think Steps ({len(recent_thinks)})")
            for i, q in enumerate(recent_thinks, 1):
                parts.append(f"{i}. {q}")

        if session.confidence_history:
            latest = session.confidence_history[-1]
            parts.append(f"\n**Latest confidence:** {latest:.0%}")

        if include_evidence and session.evidence:
            parts.append(f"\n### Evidence ({len(session.evidence)} items)")
            for ev in session.evidence[-5:]:
                src = ev.source
                snip = ev.snippet[:100]
                parts.append(f"- [{src}] {snip}")

        if include_variables:
            # List user-defined variables (not helpers)
            ns = session.repl._namespace
            helpers = set(session.repl._helpers.keys())
            builtins_keys = {"__builtins__", "ctx", "line_number_base"}
            user_vars = [
                k
                for k in ns
                if k not in helpers and k not in builtins_keys and not k.startswith("_")
            ]
            if user_vars:
                parts.append(f"\n**Variables:** {', '.join(sorted(user_vars))}")

        if clear_history:
            session.think_history.clear()
            parts.append("\n(think_history cleared)")

        return "\n".join(parts)

    async def async_summarize_so_far(
        context_id: Annotated[str, "Session identifier"] = "default",
        include_evidence: Annotated[bool, "Include evidence summary"] = True,
        include_variables: Annotated[bool, "Include REPL variable names"] = True,
        clear_history: Annotated[bool, "Clear think_history after summarizing"] = False,
    ) -> str:
        return sync_summarize_so_far(
            context_id,
            include_evidence,
            include_variables,
            clear_history,
        )

    return StructuredTool.from_function(
        name="summarize_so_far",
        description=(
            "Compress reasoning history into a summary. Useful for managing "
            "context window when analysis has many steps."
        ),
        func=sync_summarize_so_far,
        coroutine=async_summarize_so_far,
    )


def _create_get_evidence_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_get_evidence(
        context_id: Annotated[str, "Session identifier"] = "default",
        source: Annotated[str, "Filter by source: any, search, peek, exec, manual, action"] = "any",
        limit: Annotated[int, "Maximum items to return"] = 20,
        offset: Annotated[int, "Skip first N items"] = 0,
    ) -> str:
        session = manager.get_session(context_id)
        if session is None:
            return f"No session '{context_id}'."

        evidence = session.evidence
        if source != "any":
            evidence = [e for e in evidence if e.source == source]

        total = len(evidence)
        evidence = evidence[offset : offset + limit]

        if not evidence:
            return "No evidence collected yet."

        lines = [f"## Evidence ({total} total, showing {len(evidence)})"]
        for i, ev in enumerate(evidence, offset + 1):
            lines.append(f"\n### [{i}] {ev.source}")
            if ev.pattern:
                lines.append(f"Pattern: {ev.pattern}")
            if ev.line_range:
                lines.append(f"Lines: {ev.line_range[0]}-{ev.line_range[1]}")
            lines.append(f"```\n{ev.snippet[:500]}\n```")
            if ev.note:
                lines.append(f"Note: {ev.note}")

        return "\n".join(lines)

    async def async_get_evidence(
        context_id: Annotated[str, "Session identifier"] = "default",
        source: Annotated[str, "Filter by source: any, search, peek, exec, manual, action"] = "any",
        limit: Annotated[int, "Maximum items to return"] = 20,
        offset: Annotated[int, "Skip first N items"] = 0,
    ) -> str:
        return sync_get_evidence(context_id, source, limit, offset)

    return StructuredTool.from_function(
        name="get_evidence",
        description=(
            "Retrieve collected evidence/citations from the analysis. "
            "Can filter by source type and paginate."
        ),
        func=sync_get_evidence,
        coroutine=async_get_evidence,
    )


def _create_finalize_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_finalize(
        answer: Annotated[str, "The final answer with evidence citations"],
        confidence: Annotated[str, "Confidence level: high, medium, low"] = "medium",
        reasoning_summary: Annotated[str, "Brief summary of the reasoning process"] = "",
        context_id: Annotated[str, "Session identifier"] = "default",
    ) -> str:
        return _render_final_report(
            manager,
            context_id=context_id,
            answer=answer,
            confidence=confidence,
            reasoning_summary=reasoning_summary,
            completion_source="finalize",
        )

    async def async_finalize(
        answer: Annotated[str, "The final answer with evidence citations"],
        confidence: Annotated[str, "Confidence level: high, medium, low"] = "medium",
        reasoning_summary: Annotated[str, "Brief summary of the reasoning process"] = "",
        context_id: Annotated[str, "Session identifier"] = "default",
    ) -> str:
        return sync_finalize(answer, confidence, reasoning_summary, context_id)

    return StructuredTool.from_function(
        name="finalize",
        description=(
            "Mark the analysis as complete with a final evidence-backed answer. "
            "Includes confidence level and supporting evidence summary."
        ),
        func=sync_finalize,
        coroutine=async_finalize,
    )


# ===========================================================================
# Status/Meta Tools (2)
# ===========================================================================


def _create_get_status_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_get_status(
        context_id: Annotated[str, "Session identifier"] = "default",
    ) -> str:
        session = manager.get_session(context_id)
        if session is None:
            return f"No session '{context_id}'."

        # Count user variables
        ns = session.repl._namespace
        helpers = set(session.repl._helpers.keys())
        builtins_keys = {"__builtins__", "ctx", "line_number_base"}
        user_vars = [
            k for k in ns if k not in helpers and k not in builtins_keys and not k.startswith("_")
        ]
        pressure = manager.get_context_pressure_status(context_id)

        return (
            f"Context: {context_id}\n"
            f"Format: {session.meta.format.value}\n"
            f"Size: {session.meta.size_chars:,} chars, {session.meta.size_lines:,} lines\n"
            f"Tokens (est): ~{session.meta.size_tokens_estimate:,}\n"
            f"Iterations: {session.iterations}\n"
            f"Hist entries: {len(session.hist)}\n"
            f"Evidence: {len(session.evidence)} items\n"
            f"Compacted: {pressure.get('compacted')} "
            f"(count={pressure.get('compaction_count')}, "
            f"reason={pressure.get('last_compaction_reason')})\n"
            f"Pressure: token={pressure.get('token_pressure')}, "
            f"iteration={pressure.get('iteration_pressure')}, "
            f"needs_compaction={pressure.get('needs_compaction')}\n"
            f"Think steps: {len(session.think_history)}\n"
            f"Tasks: {len(session.tasks)}\n"
            f"Variables: {len(user_vars)} ({', '.join(sorted(user_vars)[:10])})"
        )

    async def async_get_status(
        context_id: Annotated[str, "Session identifier"] = "default",
    ) -> str:
        return sync_get_status(context_id)

    return StructuredTool.from_function(
        name="get_status",
        description="Get status and metadata for an RLM context session.",
        func=sync_get_status,
        coroutine=async_get_status,
    )


def _create_rlm_tasks_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_rlm_tasks(
        action: Annotated[str, "Action: list, add, update, clear"] = "list",
        context_id: Annotated[str, "Session identifier"] = "default",
        description: Annotated[str, "Task description (for add)"] = "",
        task_id: Annotated[str, "Task ID (for update)"] = "",
        status: Annotated[str, "Status: todo, done, blocked (for update)"] = "todo",
    ) -> str:
        session = manager.get_or_create_session(context_id)

        if action == "list":
            if not session.tasks:
                return "No tasks."
            lines = []
            for t in session.tasks:
                lines.append(f"[{t['id']}] [{t['status']}] {t['title']}")
            return "\n".join(lines)

        elif action == "add":
            if not description:
                return "Provide a description for the task."
            session.task_counter += 1
            task = {
                "id": session.task_counter,
                "title": description,
                "status": "todo",
                "note": None,
                "created_at": datetime.now().isoformat(),
                "updated_at": None,
            }
            session.tasks.append(task)
            return f"Task #{session.task_counter} created: {description}"

        elif action == "update":
            if not task_id:
                return "Provide a task_id to update."
            try:
                tid = int(task_id)
            except ValueError:
                return f"Invalid task_id: {task_id}"
            for t in session.tasks:
                if t["id"] == tid:
                    if status in ("todo", "done", "blocked"):
                        t["status"] = status
                    t["updated_at"] = datetime.now().isoformat()
                    return f"Task #{tid} updated to '{status}'."
            return f"Task #{tid} not found."

        elif action == "clear":
            count = len(session.tasks)
            session.tasks.clear()
            session.task_counter = 0
            return f"Cleared {count} tasks."

        return f"Unknown action: {action}"

    async def async_rlm_tasks(
        action: Annotated[str, "Action: list, add, update, clear"] = "list",
        context_id: Annotated[str, "Session identifier"] = "default",
        description: Annotated[str, "Task description (for add)"] = "",
        task_id: Annotated[str, "Task ID (for update)"] = "",
        status: Annotated[str, "Status: todo, done, blocked (for update)"] = "todo",
    ) -> str:
        return sync_rlm_tasks(action, context_id, description, task_id, status)

    return StructuredTool.from_function(
        name="rlm_tasks",
        description="Manage lightweight task tracking within an RLM session.",
        func=sync_rlm_tasks,
        coroutine=async_rlm_tasks,
    )


# ===========================================================================
# Recipe Tools (4)
# ===========================================================================


def _create_validate_recipe_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_validate_recipe(
        recipe_json: Annotated[str, "Recipe payload as JSON string"],
    ) -> str:
        from rlmagents.recipes import validate_recipe

        try:
            recipe = json.loads(recipe_json)
        except json.JSONDecodeError as exc:
            return f"Invalid JSON: {exc}"

        normalized, errors = validate_recipe(recipe)
        if errors:
            return "Validation errors:\n" + "\n".join(f"- {e}" for e in errors)
        return f"Recipe valid. {len(normalized.get('steps', []))} steps."  # type: ignore[union-attr]

    async def async_validate_recipe(
        recipe_json: Annotated[str, "Recipe payload as JSON string"],
    ) -> str:
        return sync_validate_recipe(recipe_json)

    return StructuredTool.from_function(
        name="validate_recipe",
        description="Validate an RLM recipe pipeline structure.",
        func=sync_validate_recipe,
        coroutine=async_validate_recipe,
    )


def _create_estimate_recipe_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_estimate_recipe(
        recipe_json: Annotated[str, "Recipe payload as JSON string"],
    ) -> str:
        from rlmagents.recipes import estimate_recipe, validate_recipe

        try:
            recipe = json.loads(recipe_json)
        except json.JSONDecodeError as exc:
            return f"Invalid JSON: {exc}"

        normalized, errors = validate_recipe(recipe)
        if errors:
            return "Validation errors:\n" + "\n".join(f"- {e}" for e in errors)

        estimate = estimate_recipe(normalized)  # type: ignore[arg-type]
        return json.dumps(estimate, indent=2)

    async def async_estimate_recipe(
        recipe_json: Annotated[str, "Recipe payload as JSON string"],
    ) -> str:
        return sync_estimate_recipe(recipe_json)

    return StructuredTool.from_function(
        name="estimate_recipe",
        description="Estimate cost and shape for an RLM recipe pipeline.",
        func=sync_estimate_recipe,
        coroutine=async_estimate_recipe,
    )


def _create_run_recipe_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_run_recipe(
        recipe_json: Annotated[str, "Recipe payload as JSON string"],
        context_id: Annotated[str, "Session identifier"] = "default",
    ) -> str:
        from rlmagents.recipes import validate_recipe

        try:
            recipe = json.loads(recipe_json)
        except json.JSONDecodeError as exc:
            return f"Invalid JSON: {exc}"

        normalized, errors = validate_recipe(recipe)
        if errors:
            return "Validation errors:\n" + "\n".join(f"- {e}" for e in errors)

        session = manager.get_or_create_session(context_id)
        session.iterations += 1

        # Execute recipe steps sequentially against the REPL
        steps = normalized.get("steps", [])  # type: ignore[union-attr]
        results = []
        for i, step in enumerate(steps):
            op = step.get("op", "unknown")
            try:
                result = _execute_recipe_step(session, step)
                results.append(f"Step {i + 1} ({op}): {result}")
            except Exception as exc:
                results.append(f"Step {i + 1} ({op}): ERROR - {exc}")
                break

        return "\n".join(results)

    async def async_run_recipe(
        recipe_json: Annotated[str, "Recipe payload as JSON string"],
        context_id: Annotated[str, "Session identifier"] = "default",
    ) -> str:
        return sync_run_recipe(recipe_json, context_id)

    return StructuredTool.from_function(
        name="run_recipe",
        description="Execute a validated RLM recipe pipeline against a context.",
        func=sync_run_recipe,
        coroutine=async_run_recipe,
    )


def _create_run_recipe_code_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_run_recipe_code(
        code: Annotated[str, "Recipe DSL code to compile and execute"],
        context_id: Annotated[str, "Session identifier"] = "default",
    ) -> str:
        session = manager.get_or_create_session(context_id)
        session.iterations += 1

        # Compile recipe DSL using the REPL (Recipe DSL helpers are injected)
        compile_code = f"_recipe_result = as_recipe({code})"
        result = session.repl.execute(compile_code)
        if result.error:
            return f"Recipe compile error: {result.error}"

        recipe_val = session.repl.get_variable("_recipe_result")
        if recipe_val is None:
            return "Recipe compilation produced no result."

        return f"Recipe compiled: {repr(recipe_val)[:2000]}"

    async def async_run_recipe_code(
        code: Annotated[str, "Recipe DSL code to compile and execute"],
        context_id: Annotated[str, "Session identifier"] = "default",
    ) -> str:
        return sync_run_recipe_code(code, context_id)

    return StructuredTool.from_function(
        name="run_recipe_code",
        description="Compile Recipe DSL code in the REPL and execute it.",
        func=sync_run_recipe_code,
        coroutine=async_run_recipe_code,
    )


def _execute_recipe_step(session: Any, step: dict[str, Any]) -> str:
    """Execute a single recipe step against a session's REPL."""
    op = step.get("op", "unknown")

    if op == "search":
        pattern = step.get("pattern", "")
        search_fn = session.repl.get_helper("search")
        if search_fn is None:
            return "search helper unavailable"
        results = search_fn(pattern, max_results=step.get("max_results", 50))
        session.add_evidence(
            Evidence(
                source="search",
                line_range=None,
                pattern=pattern,
                snippet=str(results[:3])[:300],
                note=f"recipe search: {len(results)} matches",
            )
        )
        return f"{len(results)} matches"

    elif op == "peek":
        start = step.get("start", 0)
        end = step.get("end", 500)
        peek_fn = session.repl.get_helper("peek")
        if peek_fn is None:
            return "peek helper unavailable"
        result = peek_fn(start, end)
        return f"{len(result)} chars"

    elif op == "chunk":
        chunk_fn = session.repl.get_helper("chunk")
        if chunk_fn is None:
            return "chunk helper unavailable"
        chunks = chunk_fn(
            chunk_size=step.get("chunk_size", 5000),
            overlap=step.get("overlap", 200),
        )
        session.chunks = [{"index": i, "size": len(c)} for i, c in enumerate(chunks)]
        return f"{len(chunks)} chunks created"

    elif op == "assign":
        name = step.get("name", "result")
        value = step.get("value", "")
        session.repl.set_variable(name, value)
        return f"assigned {name}"

    elif op == "finalize":
        return "recipe finalized"

    else:
        return f"unsupported op: {op}"


# ===========================================================================
# Config Tool (1)
# ===========================================================================


def _create_configure_rlm_tool(manager: RLMSessionManager) -> BaseTool:
    def sync_configure_rlm(
        sandbox_timeout: Annotated[float, "Sandbox timeout in seconds (0 = no change)"] = 0,
        context_policy: Annotated[
            str, "Context policy: trusted or isolated (empty = no change)"
        ] = "",
    ) -> str:
        config = manager.update_config(
            sandbox_timeout=sandbox_timeout if sandbox_timeout > 0 else None,
            context_policy=context_policy if context_policy else None,
        )
        return f"Config updated: {json.dumps(config)}"

    async def async_configure_rlm(
        sandbox_timeout: Annotated[float, "Sandbox timeout in seconds (0 = no change)"] = 0,
        context_policy: Annotated[
            str, "Context policy: trusted or isolated (empty = no change)"
        ] = "",
    ) -> str:
        return sync_configure_rlm(sandbox_timeout, context_policy)

    return StructuredTool.from_function(
        name="configure_rlm",
        description="Update RLM runtime configuration (sandbox timeout, context policy).",
        func=sync_configure_rlm,
        coroutine=async_configure_rlm,
    )
