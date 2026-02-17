"""RLMSessionManager -- manages REPL sessions without MCP dependency.

Uses rlmagents' local REPL and serialization modules.
"""

from __future__ import annotations

import asyncio
import json
import logging
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
        sub_query_max_tokens: int = 4096,
        context_token_threshold: int = 20_000,
        context_iteration_threshold: int = 100,
    ) -> None:
        self.sandbox_config = sandbox_config or SandboxConfig()
        self.context_policy = context_policy
        self._sub_query_model = sub_query_model
        self._sub_query_timeout = sub_query_timeout
        self._sub_query_max_tokens = sub_query_max_tokens
        self._context_token_threshold = context_token_threshold
        self._context_iteration_threshold = context_iteration_threshold
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
        )
        self._inject_sub_query(session, context_id)
        self.sessions[context_id] = session
        return meta

    # -- Internal helpers --------------------------------------------------

    def _refresh_session_metadata(self, session: Session) -> None:
        """Recompute context metadata from the live REPL context."""
        context_text = _coerce_context_to_text(session.repl.get_variable("ctx"))
        session.meta = _analyze_text_context(context_text, session.meta.format.value)

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
        self._inject_sub_query(session, resolved_id)
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
        self._inject_sub_query(session, resolved_id)
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

        model = self._sub_query_model
        timeout = self._sub_query_timeout

        async def _sub_query_async(prompt: str, context_slice: str | None = None) -> str:
            if model is None:
                return (
                    "[sub_query unavailable: no model configured. "
                    "Pass sub_query_model to RLMSessionManager or "
                    "create_rlm_agent to enable sub-LLM calls.]"
                )
            full_prompt = prompt
            if context_slice:
                full_prompt = f"{prompt}\n\n--- Context ---\n{context_slice}"
            try:
                response = await asyncio.wait_for(
                    model.ainvoke(full_prompt),
                    timeout=timeout,
                )
                return str(response.content)
            except asyncio.TimeoutError:
                return f"[sub_query timed out after {timeout}s]"
            except Exception as exc:
                logger.warning("sub_query error: %s", exc)
                category = _classify_sub_query_error(exc)
                return f"[sub_query error:{category}: {exc}]"

        session.repl.inject_sub_query(_sub_query_async)

    # -- Runtime config ------------------------------------------------------

    def update_config(
        self,
        *,
        sandbox_timeout: float | None = None,
        context_policy: str | None = None,
        sub_query_timeout: float | None = None,
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
        return {
            "sandbox_timeout": self.sandbox_config.timeout_seconds,
            "context_policy": self.context_policy,
            "sub_query_timeout": self._sub_query_timeout,
            "max_output_chars": self.sandbox_config.max_output_chars,
        }
