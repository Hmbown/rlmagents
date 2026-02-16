"""RLMSessionManager -- manages Aleph REPL sessions without MCP dependency.

Uses Aleph's low-level components directly: REPLEnvironment, _Session, _Evidence,
helpers, and serialization -- all of which are MCP-free.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from aleph.mcp.session import (
    _Evidence,
    _Session,
    _analyze_text_context,
    _coerce_context_to_text,
    _session_from_payload,
    _session_to_payload,
)
from aleph.mcp.workspace import DEFAULT_LINE_NUMBER_BASE, LineNumberBase, _validate_line_number_base
from aleph.repl.sandbox import REPLEnvironment, SandboxConfig
from aleph.sub_query import SubQueryConfig, detect_backend
from aleph.sub_query.cli_backend import run_cli_sub_query
from aleph.types import ContentFormat, ContextMetadata

# Re-export for convenience
__all__ = [
    "RLMSessionManager",
    "_Evidence",
    "_Session",
    "SandboxConfig",
]

# ---------------------------------------------------------------------------
# Format detection (copied from aleph.mcp.io_utils, no extra deps)
# ---------------------------------------------------------------------------

_FORMAT_CACHE_MAX = 64
_FORMAT_CACHE: OrderedDict[tuple[int, int, str], ContentFormat] = OrderedDict()


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
    """Manages Aleph REPL sessions without MCP dependency.

    Provides the same session lifecycle as ``AlephMCPServerLocal`` but uses
    Aleph's low-level components directly (REPLEnvironment, _Session, _Evidence,
    helpers), avoiding the ``mcp`` package import.
    """

    def __init__(
        self,
        sandbox_config: SandboxConfig | None = None,
        context_policy: str = "trusted",
        sub_query_config: SubQueryConfig | None = None,
    ) -> None:
        self.sandbox_config = sandbox_config or SandboxConfig()
        self.context_policy = context_policy
        self.sub_query_config = sub_query_config or SubQueryConfig()
        self.sessions: dict[str, _Session] = {}

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
        meta = _analyze_text_context(text, fmt)

        base: LineNumberBase
        try:
            base = _validate_line_number_base(line_number_base)
        except Exception:
            base = DEFAULT_LINE_NUMBER_BASE

        repl = REPLEnvironment(
            context=text,
            context_var_name="ctx",
            config=self.sandbox_config,
        )
        repl.set_variable("line_number_base", base)

        session = _Session(
            repl=repl,
            meta=meta,
            line_number_base=base,
        )
        self._inject_sub_query(session, context_id)
        self.sessions[context_id] = session
        return meta

    def get_or_create_session(self, context_id: str = "default") -> _Session:
        """Return existing session or create an empty one."""
        if context_id in self.sessions:
            return self.sessions[context_id]
        self.create_session("", context_id=context_id)
        return self.sessions[context_id]

    def get_session(self, context_id: str) -> _Session | None:
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
            }
        return result

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
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    def load_session_from_payload(
        self,
        payload: dict[str, Any],
        context_id: str | None = None,
    ) -> ContextMetadata:
        """Deserialize a session from a JSON payload."""
        resolved_id = context_id or payload.get("context_id") or payload.get("session_id") or "default"
        session = _session_from_payload(payload, resolved_id, self.sandbox_config, loop=None)
        self._inject_sub_query(session, resolved_id)
        self.sessions[resolved_id] = session
        return session.meta

    def load_session_from_file(
        self,
        path: str | Path,
        context_id: str | None = None,
    ) -> ContextMetadata:
        """Load a session from a JSON file."""
        p = Path(path)
        payload = json.loads(p.read_text(encoding="utf-8"))
        return self.load_session_from_payload(payload, context_id=context_id)

    # -- Sub-query injection -------------------------------------------------

    def _inject_sub_query(self, session: _Session, context_id: str) -> None:
        """Inject sub_query() callable into the REPL namespace.

        Follows the pattern from aleph.mcp.local_server._inject_repl_sub_query
        but uses CLI backends directly (no MCP dependency).
        """
        config = self.sub_query_config

        async def sub_query(prompt: str, context_slice: str | None = None) -> str:
            backend = config.backend
            if backend == "auto":
                backend = detect_backend(config)
            if backend == "api":
                return "[ERROR: API backend not available in rlmagents; use a CLI backend]"
            try:
                success, output = await run_cli_sub_query(
                    prompt=prompt,
                    context_slice=context_slice,
                    backend=backend,  # type: ignore[arg-type]
                    timeout=config.cli_timeout_seconds,
                    max_output_chars=config.cli_max_output_chars,
                    max_context_chars=config.max_context_chars,
                )
            except Exception as exc:
                return f"[ERROR: sub_query failed: {exc}]"
            if not success:
                return f"[ERROR: sub_query failed: {output}]"
            return output

        session.repl.inject_sub_query(sub_query)

    # -- Runtime config ------------------------------------------------------

    def update_config(
        self,
        *,
        sandbox_timeout: float | None = None,
        context_policy: str | None = None,
    ) -> dict[str, Any]:
        """Update runtime configuration. Returns the new config."""
        if sandbox_timeout is not None:
            self.sandbox_config = SandboxConfig(
                allowed_imports=self.sandbox_config.allowed_imports,
                max_output_chars=self.sandbox_config.max_output_chars,
                timeout_seconds=sandbox_timeout,
                enable_code_execution=self.sandbox_config.enable_code_execution,
                unrestricted=self.sandbox_config.unrestricted,
            )
        if context_policy is not None:
            self.context_policy = context_policy
        return {
            "sandbox_timeout": self.sandbox_config.timeout_seconds,
            "context_policy": self.context_policy,
            "max_output_chars": self.sandbox_config.max_output_chars,
        }
