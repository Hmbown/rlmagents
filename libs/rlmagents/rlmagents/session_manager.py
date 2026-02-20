"""RLMSessionManager -- manages REPL sessions without MCP dependency.

Uses rlmagents' local REPL and serialization modules.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlmagents.repl.sandbox import REPLEnvironment, SandboxConfig
from rlmagents.serialization import (
    Session,
    _analyze_text_context,
    _coerce_context_to_text,
    _session_from_payload,
    _session_to_payload,
    load_session_from_file,
    save_session_to_file,
)
from rlmagents.types import ContentFormat, ContextMetadata, Evidence

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# Re-export for convenience
__all__ = [
    "RLMSessionManager",
    "Evidence",
    "Session",
    "SandboxConfig",
]

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

_FORMAT_CACHE_MAX = 64
_FORMAT_CACHE: OrderedDict[tuple[int, int, str], ContentFormat] = OrderedDict()
_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:python|py)?\s*(?P<code>.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_FINAL_CALL_PATTERN = re.compile(r"^FINAL\((?P<value>.+)\)$", re.DOTALL)
_FINAL_VAR_CALL_PATTERN = re.compile(r"^FINAL_VAR\((?P<value>[A-Za-z_][A-Za-z0-9_]*)\)$")
_SUB_QUERY_MAX_ITERATIONS = 24
_SUB_QUERY_RECURSIVE_SYSTEM_PROMPT = (
    "You are inside a recursive RLM sub-query REPL. "
    "The context is available as `ctx`. "
    "On each turn, output Python code only. "
    "Use `search(...)`, `peek(...)`, `sub_query(...)`, and helper functions as needed. "
    "When done, set `Final` (e.g. `set_final(result)` or `Final = result`). "
    "Do not output prose."
)


def _running_loop_or_none() -> asyncio.AbstractEventLoop | None:
    """Return the active asyncio loop for this thread, if one exists."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _classify_sub_query_error(exc: Exception) -> str:
    """Classify sub_query model/provider errors for actionable retries."""
    msg = str(exc).lower()
    if any(token in msg for token in ("unauthorized", "invalid api key", "authentication", "401")):
        return "auth"
    if any(token in msg for token in ("rate limit", "429", "too many requests", "quota")):
        return "rate_limit"
    if any(token in msg for token in ("model not found", "unknown model", "404", "does not exist")):
        return "model_not_found"
    return "unknown"


def _fmt_meta(meta: ContextMetadata) -> str:
    """Format context metadata into a compact human-readable string."""
    return (
        f"Format: {meta.format.value}, "
        f"{meta.size_chars:,} chars, "
        f"{meta.size_lines:,} lines, "
        f"~{meta.size_tokens_estimate:,} tokens"
    )


def _detect_format(text: str, format_hint: str = "auto") -> ContentFormat:
    """Detect content format from text with optional hint."""
    if format_hint != "auto":
        try:
            return ContentFormat(format_hint)
        except ValueError:
            pass

    key = (hash(text[:4096]), len(text), format_hint)
    cached = _FORMAT_CACHE.get(key)
    if cached is not None:
        _FORMAT_CACHE.move_to_end(key)
        return cached

    t = text.lstrip()
    fmt = ContentFormat.TEXT
    if t.startswith("{") or t.startswith("["):
        try:
            json.loads(text)
            fmt = ContentFormat.JSON
        except Exception:
            pass

    _FORMAT_CACHE[key] = fmt
    if len(_FORMAT_CACHE) > _FORMAT_CACHE_MAX:
        _FORMAT_CACHE.popitem(last=False)
    return fmt


def _truncate_preview(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    suffix = "... [truncated]"
    keep = max(max_chars - len(suffix), 0)
    return text[:keep] + suffix


def _model_response_text(response: object) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        if parts:
            return "\n".join(parts)
    return str(content)


def _extract_python_code(text: str) -> str:
    stripped = text.strip()
    match = _FINAL_VAR_CALL_PATTERN.match(stripped)
    if match:
        return f"set_final({match.group('value')})"
    match = _FINAL_CALL_PATTERN.match(stripped)
    if match:
        return f"set_final({match.group('value')})"

    code_blocks = [m.group("code").strip() for m in _CODE_BLOCK_PATTERN.finditer(text)]
    for code in reversed(code_blocks):
        if code:
            return code
    return stripped


# ---------------------------------------------------------------------------
# RLMSessionManager
# ---------------------------------------------------------------------------


class RLMSessionManager:
    """Manages REPL sessions without MCP dependency.

    Provides session lifecycle management for RLM context isolation.
    Optionally accepts a LangChain model for sub_query support
    (Algorithm 1 from the RLM paper).
    """

    def __init__(
        self,
        sandbox_config: SandboxConfig | None = None,
        context_policy: str = "trusted",
        sub_query_model: BaseChatModel | None = None,
        sub_query_timeout: float = 120.0,
        rlm_max_recursion_depth: int = 1,
        sub_query_max_tokens: int = 4096,
        context_token_threshold: int = 20_000,
        context_iteration_threshold: int = 100,
        hist_max_entries: int = 64,
        hist_max_code_chars: int = 280,
        hist_max_text_chars: int = 200,
        enable_final_sentinel: bool = False,
    ) -> None:
        self.sandbox_config = sandbox_config or SandboxConfig()
        self.context_policy = context_policy
        self._sub_query_model = sub_query_model
        self._sub_query_timeout = sub_query_timeout
        self._rlm_max_recursion_depth = max(0, int(rlm_max_recursion_depth))
        self._sub_query_max_tokens = sub_query_max_tokens
        self._context_token_threshold = context_token_threshold
        self._context_iteration_threshold = context_iteration_threshold
        self._hist_max_entries = max(1, hist_max_entries)
        self._hist_max_code_chars = max(40, hist_max_code_chars)
        self._hist_max_text_chars = max(40, hist_max_text_chars)
        self.enable_final_sentinel = enable_final_sentinel
        self._loop = _running_loop_or_none()
        self.sessions: dict[str, Session] = {}

    # -- Sub-query model configuration ---------------------------------------

    def set_sub_query_model(self, model: BaseChatModel) -> None:
        """Set or update the model used for sub_query calls."""
        self._sub_query_model = model
        # Re-inject into all existing sessions
        for cid, session in self.sessions.items():
            self._inject_sub_query(session, cid)

    def set_loop(self, loop: asyncio.AbstractEventLoop | None) -> None:
        """Set/replace the event loop used by REPL sub_query bridges."""
        self._loop = loop
        for session in self.sessions.values():
            session.repl.set_loop(loop)

    # -- Session lifecycle ---------------------------------------------------

    def create_session(
        self,
        content: str | Any,
        context_id: str = "default",
        format_hint: str = "auto",
        line_number_base: int = 1,
    ) -> ContextMetadata:
        """Create a new session with the given content.

        If a session with ``context_id`` already exists, it is replaced.
        """
        text = _coerce_context_to_text(content)
        fmt = _detect_format(text, format_hint)
        meta = _analyze_text_context(text, fmt.value)

        repl = REPLEnvironment(
            context=text,
            context_var_name="ctx",
            config=self.sandbox_config,
        )
        repl.set_variable("line_number_base", line_number_base)

        session = Session(
            repl=repl,
            meta=meta,
            line_number_base=line_number_base,
            context_token_threshold=self._context_token_threshold,
            context_iteration_threshold=self._context_iteration_threshold,
            max_hist_entries=self._hist_max_entries,
            max_hist_code_chars=self._hist_max_code_chars,
            max_hist_text_chars=self._hist_max_text_chars,
        )
        self._inject_sub_query(session, context_id)
        self._sync_session_runtime_vars(session)
        self.sessions[context_id] = session
        return meta

    # -- Internal helpers --------------------------------------------------

    def _refresh_session_metadata(self, session: Session) -> None:
        """Recompute context metadata from the live REPL context."""
        context_text = _coerce_context_to_text(session.repl.get_variable("ctx"))
        session.meta = _analyze_text_context(context_text, session.meta.format.value)

    def _sync_session_runtime_vars(self, session: Session) -> None:
        """Mirror manager-maintained artifacts into the REPL namespace."""
        session.repl.set_variable("hist", [dict(item) for item in session.hist])
        if session.repl.get_variable("Final") is None:
            session.repl.set_variable("Final", None)

    # -- Context-pressure policy --------------------------------------------

    def summarize_so_far(
        self,
        context_id: str,
        include_evidence: bool = True,
        include_variables: bool = True,
        clear_history: bool = False,
    ) -> str:
        """Generate a compact session summary with optional history and evidence."""
        session = self.get_session(context_id)
        if session is None:
            return f"No session '{context_id}'."

        parts = [f"## Session Summary: {context_id}"]
        parts.append(f"Iterations: {session.iterations}")
        parts.append(f"Context: {_fmt_meta(session.meta)}")

        if session.think_history:
            recent_thinks = session.think_history[-10:]
            parts.append(f"\n### Think Steps ({len(recent_thinks)})")
            for i, q in enumerate(recent_thinks, 1):
                parts.append(f"{i}. {q}")

        if session.confidence_history:
            latest = session.confidence_history[-1]
            parts.append(f"\n### Confidence: {latest:.0%}")

        if include_evidence and session.evidence:
            parts.append(f"\n### Evidence ({len(session.evidence)} items)")
            for ev in session.evidence[-5:]:
                src = ev.source
                op = ev.source_op or "tool"
                snip = ev.snippet[:100]
                parts.append(f"- [{src}:{op}] {snip}")

        if include_variables:
            ns = session.repl._namespace
            helpers = set(session.repl._helpers.keys())
            builtins_keys = {"__builtins__", "ctx", "line_number_base"}
            user_vars = [
                k
                for k in ns
                if k not in helpers and k not in builtins_keys and not k.startswith("_")
            ]
            if user_vars:
                parts.append(f"\nVariables: {', '.join(sorted(user_vars))}")

        summary = "\n".join(parts)
        if clear_history:
            session.think_history.clear()
        return summary

    def _needs_context_compaction(self, session: Session) -> bool:
        """Return True if session has exceeded pressure thresholds."""
        token_threshold = session.context_token_threshold
        iteration_threshold = session.context_iteration_threshold
        if not iteration_threshold and not token_threshold:
            return False

        if iteration_threshold > 0:
            if session.iterations >= session.last_compaction_iteration + iteration_threshold:
                return True

        if token_threshold > 0:
            if session.meta.size_tokens_estimate >= token_threshold:
                return True

        return False

    def _compact_preserve_snippets(self, session: Session, summary: str) -> set[str]:
        """Collect evidence snippets to keep during compaction pruning."""
        preserve: set[str] = {summary}
        for ev in session.evidence:
            if session._is_high_signal_evidence(ev):
                preserve.add(ev.snippet)
        return preserve

    def _compact_session(self, context_id: str, session: Session) -> str:
        """Compact context by synthesizing a summary and pruning evidence."""
        summary = self.summarize_so_far(context_id, include_variables=False)
        session.compaction_count += 1
        reason = "manual"
        if (
            session.context_iteration_threshold > 0
            and session.iterations >= session.context_iteration_threshold
        ):
            reason = "iterations"
        elif (
            session.context_token_threshold > 0
            and session.meta.size_tokens_estimate >= session.context_token_threshold
        ):
            reason = "tokens"
        session.last_compaction_iteration = session.iterations
        session.compacted_state = summary
        session.compacted_state_at = datetime.now().isoformat()
        session.last_compaction_reason = reason
        preserve_snippets = self._compact_preserve_snippets(session, summary)
        session.add_evidence(
            Evidence(
                source="manual",
                source_op="summarize_so_far",
                context_id=context_id,
                pattern="auto_compaction",
                snippet=summary,
                note="Auto summary added by context pressure policy",
            ),
            preserve_snippets=preserve_snippets,
        )
        return summary

    def _pressure_status_payload(self, session: Session) -> dict[str, object]:
        """Build compacted-state pressure metadata for reporting."""
        iteration_threshold = session.context_iteration_threshold
        token_threshold = session.context_token_threshold
        needs_compaction = self._needs_context_compaction(session)
        iteration_pressure = (
            session.iterations / iteration_threshold if iteration_threshold else 0.0
        )
        token_pressure = (
            session.meta.size_tokens_estimate / token_threshold if token_threshold else 0.0
        )

        return {
            "compacted": session.compaction_count > 0,
            "compaction_count": session.compaction_count,
            "last_compaction_reason": session.last_compaction_reason,
            "last_compaction_iteration": session.last_compaction_iteration,
            "last_compaction_at": session.compacted_state_at,
            "has_compacted_state": session.compacted_state is not None,
            "iteration_threshold": iteration_threshold,
            "token_threshold": token_threshold,
            "current_iteration": session.iterations,
            "current_tokens": session.meta.size_tokens_estimate,
            "needs_compaction": needs_compaction,
            "iteration_pressure": iteration_pressure,
            "token_pressure": token_pressure,
        }

    def _note_tool_activity(self, context_id: str, source_op: str | None = None) -> str | None:
        """Record tool activity and return an auto-compaction summary when triggered."""
        session = self.get_session(context_id)
        if session is None:
            return None
        session.iterations += 1
        self._refresh_session_metadata(session)
        if not self._needs_context_compaction(session):
            return None
        if source_op:
            session.think_history.append(f"Auto-pressure event triggered: {source_op}")
        return self._compact_session(context_id, session)

    def append_hist_entry(
        self,
        context_id: str,
        entry: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Append a bounded root-loop history entry for a context."""
        session = self.get_session(context_id)
        if session is None:
            return None
        normalized = session.append_hist(entry)
        self._sync_session_runtime_vars(session)
        return normalized

    def get_recent_hist_entries(
        self,
        *,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        """Return recent history entries across sessions (newest first)."""
        if limit <= 0:
            return []
        entries: list[dict[str, Any]] = []
        for context_id, session in self.sessions.items():
            for item in session.hist:
                enriched = dict(item)
                enriched.setdefault("context_id", context_id)
                entries.append(enriched)

        def _sort_key(item: dict[str, Any]) -> tuple[str, int]:
            timestamp = item.get("timestamp")
            ts = timestamp if isinstance(timestamp, str) else ""
            iteration = item.get("iteration")
            it = int(iteration) if isinstance(iteration, int) else 0
            return (ts, it)

        entries.sort(key=_sort_key, reverse=True)
        return entries[:limit]

    def format_recent_exec_metadata(
        self,
        *,
        limit: int = 4,
        max_chars: int = 1200,
    ) -> str:
        """Build a bounded root-loop execution metadata summary for model context."""
        entries = self.get_recent_hist_entries(limit=limit)
        if not entries:
            return ""

        lines = ["[RLM per-iteration execution metadata]"]
        for item in entries:
            context_id = str(item.get("context_id", "default"))
            iteration = item.get("iteration", "?")
            status = "error" if item.get("error") else "ok"
            stdout_chars = int(item.get("stdout_chars") or 0)
            stderr_chars = int(item.get("stderr_chars") or 0)
            duration = item.get("execution_time_ms")
            duration_label = (
                f"{float(duration):.1f}ms"
                if isinstance(duration, (int, float))
                else "n/a"
            )
            vars_updated = int(item.get("variables_updated_count") or 0)
            final_set = bool(item.get("final_set"))
            code_preview = str(item.get("code_preview") or "")
            line = (
                f"- {context_id}#{iteration} exec_python {status}; "
                f"stdout={stdout_chars}, stderr={stderr_chars}, vars={vars_updated}, "
                f"t={duration_label}, final={'set' if final_set else 'unset'}; "
                f"code={code_preview!r}"
            )
            lines.append(line)

        text = "\n".join(lines)
        if max_chars <= 0 or len(text) <= max_chars:
            return text
        suffix = "\n... [metadata truncated]"
        keep = max(max_chars - len(suffix), 0)
        return text[:keep] + suffix

    def consume_final_sentinel(self, context_id: str) -> str | None:
        """Return and clear REPL `Final` sentinel value for a context."""
        session = self.get_session(context_id)
        if session is None:
            return None
        final_value = session.repl.get_variable("Final")
        if final_value is None:
            return None
        session.repl.set_variable("Final", None)
        return _coerce_context_to_text(final_value)

    def get_context_pressure_status(self, context_id: str) -> dict[str, object]:
        """Return compacted-state metadata for context status reporting."""
        session = self.get_session(context_id)
        if session is None:
            return {}

        status = self._pressure_status_payload(session)
        status["compacted_state"] = session.compacted_state is not None
        return status

    def get_or_create_session(self, context_id: str = "default") -> Session:
        """Return existing session or create an empty one."""
        if context_id in self.sessions:
            return self.sessions[context_id]
        self.create_session("", context_id=context_id)
        return self.sessions[context_id]

    def get_session(self, context_id: str) -> Session | None:
        """Return a session by ID, or None if it doesn't exist."""
        return self.sessions.get(context_id)

    def delete_session(self, context_id: str) -> bool:
        """Delete a session. Returns True if it existed."""
        return self.sessions.pop(context_id, None) is not None

    def list_sessions(self) -> dict[str, dict[str, Any]]:
        """Return summary metadata for all sessions."""
        result: dict[str, dict[str, Any]] = {}
        for cid, session in self.sessions.items():
            result[cid] = {
                "format": session.meta.format.value,
                "size_chars": session.meta.size_chars,
                "size_lines": session.meta.size_lines,
                "size_tokens_estimate": session.meta.size_tokens_estimate,
                "iterations": session.iterations,
                "evidence_count": len(session.evidence),
                "hist_count": len(session.hist),
                "task_count": len(session.tasks),
                "compaction_count": session.compaction_count,
                "context_pressure": {
                    "iteration": session.iterations,
                    "iteration_threshold": session.context_iteration_threshold,
                    "token_estimate": session.meta.size_tokens_estimate,
                    "token_threshold": session.context_token_threshold,
                },
                "pressure_status": self.get_context_pressure_status(cid),
            }
        return result

    def get_pressure_status(self, context_id: str) -> dict[str, object]:
        """Return compacted-state metadata for context status reporting."""
        status = self.get_context_pressure_status(context_id)
        if status:
            status["has_compacted_state"] = bool(status.get("has_compacted_state"))
            return status

        if (session := self.get_session(context_id)) is None:
            return {
                "compacted": False,
                "compaction_count": 0,
                "has_compacted_state": False,
            }

        return {
            "compacted": False,
            "compaction_count": 0,
            "has_compacted_state": False,
            "iteration_threshold": session.context_iteration_threshold,
            "token_threshold": session.context_token_threshold,
            "current_iteration": session.iterations,
            "current_tokens": session.meta.size_tokens_estimate,
            "needs_compaction": self._needs_context_compaction(session),
            "iteration_pressure": session.iterations / max(session.context_iteration_threshold, 1),
            "token_pressure": session.meta.size_tokens_estimate / max(session.context_token_threshold, 1),
        }

    # -- Serialization -------------------------------------------------------

    def save_session(
        self,
        context_id: str = "default",
        path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Serialize session to JSON-safe dict, optionally writing to file."""
        session = self.sessions.get(context_id)
        if session is None:
            raise KeyError(f"No session with context_id={context_id!r}")
        payload = _session_to_payload(context_id, session, include_ctx=True)
        if path is not None:
            save_session_to_file(session, context_id, path, include_ctx=True)
        return payload

    def load_session_from_payload(
        self,
        payload: dict[str, Any],
        context_id: str | None = None,
    ) -> ContextMetadata:
        """Deserialize a session from a JSON payload."""
        resolved_id = (
            context_id or payload.get("context_id") or payload.get("session_id") or "default"
        )
        session = _session_from_payload(payload, resolved_id, self.sandbox_config)
        session.context_token_threshold = self._context_token_threshold
        session.context_iteration_threshold = self._context_iteration_threshold
        session.max_hist_entries = self._hist_max_entries
        session.max_hist_code_chars = self._hist_max_code_chars
        session.max_hist_text_chars = self._hist_max_text_chars
        session._prune_hist()
        self._inject_sub_query(session, resolved_id)
        self._sync_session_runtime_vars(session)
        self.sessions[resolved_id] = session
        return session.meta

    def load_session_from_file(
        self,
        path: str | Path,
        context_id: str | None = None,
    ) -> ContextMetadata:
        """Load a session from a JSON file."""
        resolved_id, session = load_session_from_file(path, self.sandbox_config, context_id)
        session.context_token_threshold = self._context_token_threshold
        session.context_iteration_threshold = self._context_iteration_threshold
        session.max_hist_entries = self._hist_max_entries
        session.max_hist_code_chars = self._hist_max_code_chars
        session.max_hist_text_chars = self._hist_max_text_chars
        session._prune_hist()
        self._inject_sub_query(session, resolved_id)
        self._sync_session_runtime_vars(session)
        self.sessions[resolved_id] = session
        return session.meta

    def _get_pressure_status(self, session: Session) -> str:
        """Return human-readable context-pressure status."""
        if session.compaction_count <= 0:
            return "not_compacted"
        reason = session.last_compaction_reason or "unknown"
        return f"compacted:{reason}#{session.compaction_count}"

    # -- Sub-query injection (Algorithm 1 core) ------------------------------

    def _inject_sub_query(self, session: Session, context_id: str) -> None:
        """Inject sub_query() and llm_query() into the REPL namespace.

        This is the core mechanism from Algorithm 1 of the RLM paper:
        the REPL environment gets a function that can invoke a sub-LLM,
        enabling programmatic recursion (e.g. looping over chunks and
        querying the LLM for each).
        """
        if self._loop is None:
            self._loop = _running_loop_or_none()
        session.repl.set_loop(self._loop)

        async def _flat_sub_query_async(full_prompt: str) -> str:
            model = self._sub_query_model
            if model is None:
                return (
                    "[sub_query unavailable: no model configured. "
                    "Pass sub_query_model to RLMSessionManager or "
                    "create_rlm_agent to enable sub-LLM calls.]"
                )
            try:
                response = await asyncio.wait_for(
                    model.ainvoke(full_prompt),
                    timeout=self._sub_query_timeout,
                )
                return _model_response_text(response)
            except asyncio.TimeoutError:
                return f"[sub_query timed out after {self._sub_query_timeout}s]"
            except Exception as exc:
                logger.warning("sub_query error: %s", exc)
                category = _classify_sub_query_error(exc)
                return f"[sub_query error:{category}: {exc}]"

        async def _run_sub_rlm_loop(
            full_prompt: str,
            *,
            depth: int,
            current_context_id: str,
        ) -> str:
            model = self._sub_query_model
            if model is None:
                return (
                    "[sub_query unavailable: no model configured. "
                    "Pass sub_query_model to RLMSessionManager or "
                    "create_rlm_agent to enable sub-LLM calls.]"
                )

            if depth >= self._rlm_max_recursion_depth:
                return await _flat_sub_query_async(full_prompt)

            loop = self._loop or _running_loop_or_none()
            child_repl = REPLEnvironment(
                context=full_prompt,
                context_var_name="ctx",
                config=self.sandbox_config,
                loop=loop,
            )
            child_repl.set_variable("line_number_base", 1)
            child_repl.set_variable("Final", None)

            async def _nested_sub_query_async(
                nested_prompt: str,
                nested_context_slice: str | None = None,
            ) -> str:
                nested_full_prompt = nested_prompt
                if nested_context_slice:
                    nested_full_prompt = f"{nested_prompt}\n\n--- Context ---\n{nested_context_slice}"
                return await _run_sub_rlm_loop(
                    nested_full_prompt,
                    depth=depth + 1,
                    current_context_id=current_context_id,
                )

            child_repl.inject_sub_query(_nested_sub_query_async)

            meta = _analyze_text_context(full_prompt, "text")
            context_preview = _truncate_preview(full_prompt, self._hist_max_text_chars)
            hist: list[str] = [
                (
                    f"[sub-context] depth={depth} context_id={current_context_id} "
                    f"chars={meta.size_chars:,} lines={meta.size_lines:,} "
                    f"tokens~={meta.size_tokens_estimate:,} "
                    f"prefix={context_preview!r}"
                )
            ]
            last_observation = ""
            max_iterations = max(
                1,
                min(self._context_iteration_threshold, _SUB_QUERY_MAX_ITERATIONS),
            )

            for iteration in range(1, max_iterations + 1):
                history_text = "\n\n".join(hist[-(self._hist_max_entries * 2) :])
                model_prompt = (
                    f"{_SUB_QUERY_RECURSIVE_SYSTEM_PROMPT}\n\n"
                    f"Depth: {depth} (max recursion depth: {self._rlm_max_recursion_depth})\n"
                    f"Iteration: {iteration}/{max_iterations}\n"
                    f"History:\n{history_text}\n\n"
                    "Return only executable Python code for the next REPL step."
                )
                try:
                    response = await asyncio.wait_for(
                        model.ainvoke(model_prompt),
                        timeout=self._sub_query_timeout,
                    )
                except asyncio.TimeoutError:
                    return f"[sub_query timed out after {self._sub_query_timeout}s]"
                except Exception as exc:
                    logger.warning("sub_query error: %s", exc)
                    category = _classify_sub_query_error(exc)
                    return f"[sub_query error:{category}: {exc}]"

                response_text = _model_response_text(response)
                code = _extract_python_code(response_text)
                if not code:
                    return "[sub_query error:empty_code]"

                result = await child_repl.execute_async(code)
                stdout_preview = _truncate_preview(result.stdout, self._hist_max_text_chars)
                stderr_preview = _truncate_preview(result.stderr, self._hist_max_text_chars)
                error_preview = _truncate_preview(result.error or "", self._hist_max_text_chars)
                code_preview = _truncate_preview(code, self._hist_max_code_chars)

                hist.append(f"[code #{iteration}]\n{code_preview}")
                hist.append(
                    (
                        f"[exec #{iteration}] updated={result.variables_updated} "
                        f"stdout_len={len(result.stdout)} stderr_len={len(result.stderr)} "
                        f"error={'yes' if result.error else 'no'}\n"
                        f"stdout={stdout_preview!r}\n"
                        f"stderr={stderr_preview!r}\n"
                        f"exec_error={error_preview!r}"
                    )
                )

                final_value = child_repl.get_variable("Final")
                if final_value is not None:
                    return _coerce_context_to_text(final_value)

                if result.stdout:
                    last_observation = result.stdout
                elif result.error:
                    last_observation = f"[sub_query exec error: {result.error}]"
                elif result.stderr:
                    last_observation = result.stderr
                elif result.return_value is not None:
                    last_observation = _coerce_context_to_text(result.return_value)

            if last_observation:
                return _truncate_preview(last_observation, self.sandbox_config.max_output_chars)
            return "[sub_query reached max iterations without Final]"

        async def _sub_query_async(prompt: str, context_slice: str | None = None) -> str:
            full_prompt = prompt
            if context_slice:
                full_prompt = f"{prompt}\n\n--- Context ---\n{context_slice}"
            return await _run_sub_rlm_loop(
                full_prompt,
                depth=0,
                current_context_id=context_id,
            )

        session.repl.inject_sub_query(_sub_query_async)

    # -- Runtime config ------------------------------------------------------

    def update_config(
        self,
        *,
        sandbox_timeout: float | None = None,
        context_policy: str | None = None,
        sub_query_timeout: float | None = None,
        rlm_max_recursion_depth: int | None = None,
    ) -> dict[str, Any]:
        """Update runtime configuration. Returns the new config."""
        if sandbox_timeout is not None:
            self.sandbox_config = SandboxConfig(
                allowed_imports=self.sandbox_config.allowed_imports,
                max_output_chars=self.sandbox_config.max_output_chars,
                timeout_seconds=sandbox_timeout,
                enable_code_execution=(self.sandbox_config.enable_code_execution),
                unrestricted=self.sandbox_config.unrestricted,
            )
        if context_policy is not None:
            self.context_policy = context_policy
        if sub_query_timeout is not None:
            self._sub_query_timeout = sub_query_timeout
        if rlm_max_recursion_depth is not None:
            self._rlm_max_recursion_depth = max(0, int(rlm_max_recursion_depth))
        return {
            "sandbox_timeout": self.sandbox_config.timeout_seconds,
            "context_policy": self.context_policy,
            "sub_query_timeout": self._sub_query_timeout,
            "rlm_max_recursion_depth": self._rlm_max_recursion_depth,
            "max_output_chars": self.sandbox_config.max_output_chars,
        }
