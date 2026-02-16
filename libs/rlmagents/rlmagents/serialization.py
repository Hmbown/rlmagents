"""Session serialization for rlmagents.

This module provides session serialization and deserialization for memory-pack
functionality.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rlmagents.repl.sandbox import REPLEnvironment, SandboxConfig
from rlmagents.types import (
    _VALID_EVIDENCE_SOURCES,
    ContentFormat,
    ContextMetadata,
    Evidence,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEMORY_PACK_RELATIVE_PATH = ".rlm/memory_pack.json"


# ---------------------------------------------------------------------------
# Session class
# ---------------------------------------------------------------------------


class Session:
    """Session state for a context."""

    def __init__(
        self,
        repl: REPLEnvironment,
        meta: ContextMetadata,
        line_number_base: int = 1,
        created_at: datetime | None = None,
        iterations: int = 0,
        think_history: list[str] | None = None,
        evidence: list[Evidence] | None = None,
        confidence_history: list[float] | None = None,
        information_gain: list[int] | None = None,
        chunks: list[dict] | None = None,
        tasks: list[dict[str, Any]] | None = None,
        task_counter: int = 0,
        max_evidence: int = 100,
        context_token_threshold: int = 20_000,
        context_iteration_threshold: int = 100,
        compaction_count: int = 0,
        last_compaction_iteration: int = 0,
        last_compaction_reason: str | None = None,
        compacted_state: str | None = None,
        compacted_state_at: str | None = None,
    ) -> None:
        self.repl = repl
        self.meta = meta
        self.line_number_base = line_number_base
        self.created_at = created_at or datetime.now()
        self.iterations = iterations
        self.think_history = think_history or []
        self.evidence = evidence or []
        self.confidence_history = confidence_history or []
        self.information_gain = information_gain or []
        self.chunks = chunks
        self.tasks = tasks or []
        self.task_counter = task_counter
        self.max_evidence = max_evidence
        self.context_token_threshold = context_token_threshold
        self.context_iteration_threshold = context_iteration_threshold
        self.compaction_count = compaction_count
        self.last_compaction_iteration = last_compaction_iteration
        self.last_compaction_reason = last_compaction_reason
        self.compacted_state = compacted_state
        self.compacted_state_at = compacted_state_at

    def add_evidence(self, ev: Evidence, preserve_snippets: set[str] | None = None) -> None:
        """Add evidence and prune if evidence limits are exceeded."""
        self.evidence.append(ev)
        self._prune_evidence(preserve_snippets)

    @staticmethod
    def _is_high_signal_evidence(ev: Evidence) -> bool:
        """Return whether evidence should be preserved during aggressive pruning."""
        if ev.source in {"exec", "sub_query", "cross_context_search", "manual"}:
            return True
        if ev.command_exit_status is not None and ev.command_exit_status != 0:
            return True
        if ev.file_path:
            return True
        return False

    def _prune_evidence(self, preserve_snippets: set[str] | None = None) -> None:
        """Prune evidence while preserving recent and high-signal records."""
        if len(self.evidence) <= self.max_evidence:
            return

        preserve_snippets = preserve_snippets or set()
        preserve_by_snippet: set[int] = {
            idx
            for idx, ev in enumerate(self.evidence)
            if ev.snippet in preserve_snippets
        }

        high_signal_indices: list[int] = [
            idx for idx, ev in enumerate(self.evidence) if self._is_high_signal_evidence(ev)
        ]
        protected_indices = set(preserve_by_snippet)
        recent_window = max(5, min(20, self.max_evidence // 2))
        protected_indices.update(
            range(
                max(0, len(self.evidence) - recent_window),
                len(self.evidence),
            )
        )
        for idx in high_signal_indices:
            protected_indices.add(idx)

        protected: list[tuple[int, Evidence]] = [
            (idx, self.evidence[idx]) for idx in sorted(protected_indices)
        ]

        cap = self.max_evidence
        if len(protected) > cap:
            protected = protected[-cap:]

        protected_lookup = {idx for idx, _ in protected}
        remaining_slots = cap - len(protected)
        unprotected: list[tuple[int, Evidence]] = [
            (idx, ev)
            for idx, ev in enumerate(self.evidence)
            if idx not in protected_lookup
        ]
        if remaining_slots <= 0:
            kept_unprotected: list[tuple[int, Evidence]] = []
        else:
            kept_unprotected = unprotected[-remaining_slots:]

        merged = sorted(
            list(protected) + kept_unprotected,
            key=lambda item: item[0],
        )
        self.evidence = [ev for _, ev in merged]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_context_to_text(value: Any) -> str:
    """Coerce various context types to text."""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except Exception:
            return str(value)
    return str(value)


def _analyze_text_context(text: str, fmt: str = "auto") -> ContextMetadata:
    """Analyze text and return metadata."""
    format_enum = ContentFormat.TEXT
    if fmt != "auto":
        try:
            format_enum = ContentFormat(fmt)
        except ValueError:
            pass
    elif text.lstrip().startswith("{") or text.lstrip().startswith("["):
        try:
            json.loads(text)
            format_enum = ContentFormat.JSON
        except Exception:
            pass

    return ContextMetadata(
        format=format_enum,
        size_bytes=len(text.encode("utf-8", errors="ignore")),
        size_chars=len(text),
        size_lines=text.count("\n") + 1,
        size_tokens_estimate=len(text) // 4,
        structure_hint=None,
        sample_preview=text[:500],
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _session_to_payload(
    session_id: str,
    session: Session,
    *,
    include_ctx: bool = True,
) -> dict[str, Any]:
    """Serialize a session to a JSON-safe dict."""
    ctx_val = session.repl.get_variable("ctx")
    ctx_text = _coerce_context_to_text(ctx_val)

    tasks_payload: list[dict[str, Any]] = []
    for task in session.tasks:
        if isinstance(task, dict):
            tasks_payload.append(task)

    payload: dict[str, Any] = {
        "schema": "rlm.session.v1",
        "session_id": session_id,
        "context_id": session_id,
        "created_at": session.created_at.isoformat(),
        "iterations": session.iterations,
        "line_number_base": session.line_number_base,
        "meta": {
            "format": session.meta.format.value,
            "size_bytes": session.meta.size_bytes,
            "size_chars": session.meta.size_chars,
            "size_lines": session.meta.size_lines,
            "size_tokens_estimate": session.meta.size_tokens_estimate,
            "structure_hint": session.meta.structure_hint,
            "sample_preview": session.meta.sample_preview,
        },
        "think_history": list(session.think_history),
        "confidence_history": list(session.confidence_history),
        "information_gain": list(session.information_gain),
        "chunks": session.chunks,
        "tasks": tasks_payload,
        "task_counter": session.task_counter,
        "context_pressure": {
            "context_token_threshold": session.context_token_threshold,
            "context_iteration_threshold": session.context_iteration_threshold,
            "compaction_count": session.compaction_count,
            "last_compaction_iteration": session.last_compaction_iteration,
            "last_compaction_reason": session.last_compaction_reason,
            "compacted_state": session.compacted_state,
            "compacted_state_at": session.compacted_state_at,
        },
        "evidence": [
            {
                "source": ev.source,
                "source_op": ev.source_op,
                "context_id": ev.context_id,
                "file_path": ev.file_path,
                "line_range": list(ev.line_range) if ev.line_range else None,
                "pattern": ev.pattern,
                "snippet": ev.snippet,
                "note": ev.note,
                "command_exit_status": ev.command_exit_status,
                "timestamp": ev.timestamp,
            }
            for ev in session.evidence
        ],
    }
    if include_ctx:
        payload["ctx"] = ctx_text
    else:
        payload["ctx_redacted"] = True
        payload["ctx_chars"] = len(ctx_text)
    return payload


def _session_from_payload(
    obj: dict[str, Any],
    resolved_id: str,
    sandbox_config: SandboxConfig,
) -> Session:
    """Deserialize a session from a JSON payload."""
    ctx = obj.get("ctx")
    if not isinstance(ctx, str):
        raise ValueError("Invalid session payload: ctx must be a string")

    meta_obj = obj.get("meta")
    if not isinstance(meta_obj, dict):
        meta_obj = {}

    try:
        fmt = ContentFormat(str(meta_obj.get("format") or "text"))
    except Exception:
        fmt = ContentFormat.TEXT

    meta = ContextMetadata(
        format=fmt,
        size_bytes=int(meta_obj.get("size_bytes") or len(ctx.encode("utf-8", errors="ignore"))),
        size_chars=int(meta_obj.get("size_chars") or len(ctx)),
        size_lines=int(meta_obj.get("size_lines") or (ctx.count("\n") + 1)),
        size_tokens_estimate=int(meta_obj.get("size_tokens_estimate") or (len(ctx) // 4)),
        structure_hint=meta_obj.get("structure_hint"),
        sample_preview=str(meta_obj.get("sample_preview") or ctx[:500]),
    )

    repl = REPLEnvironment(
        context=ctx,
        context_var_name="ctx",
        config=sandbox_config,
    )
    raw_line_number_base = obj.get("line_number_base")
    if isinstance(raw_line_number_base, (int, str)):
        line_number_base_val = raw_line_number_base
    else:
        line_number_base_val = 1
    try:
        base = int(line_number_base_val)
    except Exception:
        base = 1
    repl.set_variable("line_number_base", base)

    created_at = datetime.now()
    created_at_str = obj.get("created_at")
    if isinstance(created_at_str, str):
        try:
            created_at = datetime.fromisoformat(created_at_str)
        except Exception:
            pass

    tasks_payload = obj.get("tasks")
    tasks: list[dict[str, Any]] = []
    if isinstance(tasks_payload, list):
        for task in tasks_payload:
            if not isinstance(task, dict):
                continue
            if "id" not in task or "title" not in task:
                continue
            raw_id = task.get("id")
            if raw_id is None:
                continue
            try:
                task_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            tasks.append(
                {
                    "id": task_id,
                    "title": str(task.get("title")),
                    "status": str(task.get("status") or "todo"),
                    "note": task.get("note"),
                    "created_at": task.get("created_at"),
                    "updated_at": task.get("updated_at"),
                }
            )

    raw_task_counter = obj.get("task_counter")
    if isinstance(raw_task_counter, (int, str)):
        try:
            task_counter = int(raw_task_counter)
        except (TypeError, ValueError):
            task_counter = max((t["id"] for t in tasks), default=0)
    else:
        task_counter = max((t["id"] for t in tasks), default=0)

    context_pressure = obj.get("context_pressure", {})
    if not isinstance(context_pressure, dict):
        context_pressure = {}

    session = Session(
        repl=repl,
        meta=meta,
        line_number_base=base,
        created_at=created_at,
        iterations=int(obj.get("iterations") or 0),
        think_history=list(obj.get("think_history") or []),
        confidence_history=list(obj.get("confidence_history") or []),
        information_gain=list(obj.get("information_gain") or []),
        chunks=obj.get("chunks"),
        tasks=tasks,
        task_counter=task_counter,
        context_token_threshold=int(context_pressure.get("context_token_threshold") or 20_000),
        context_iteration_threshold=int(
            context_pressure.get("context_iteration_threshold") or 100
        ),
        compaction_count=int(context_pressure.get("compaction_count") or 0),
        last_compaction_iteration=int(context_pressure.get("last_compaction_iteration") or 0),
        last_compaction_reason=context_pressure.get("last_compaction_reason"),
        compacted_state=context_pressure.get("compacted_state"),
        compacted_state_at=context_pressure.get("compacted_state_at"),
    )

    ev_list = obj.get("evidence")
    if isinstance(ev_list, list):
        for ev in ev_list:
            if not isinstance(ev, dict):
                continue
            source = ev.get("source")
            if source not in _VALID_EVIDENCE_SOURCES:
                continue
            line_range = ev.get("line_range")
            if isinstance(line_range, list) and len(line_range) == 2:
                try:
                    line_range = (int(line_range[0]), int(line_range[1]))
                except Exception:
                    line_range = None
            else:
                line_range = None
            session.evidence.append(
                Evidence(
                    source=source,  # type: ignore[arg-type]
                    source_op=ev.get("source_op"),
                    context_id=ev.get("context_id"),
                    file_path=ev.get("file_path"),
                    line_range=line_range,
                    pattern=ev.get("pattern"),
                    snippet=ev.get("snippet", ""),
                    note=ev.get("note"),
                    command_exit_status=(
                        int(ev.get("command_exit_status"))
                        if isinstance(ev.get("command_exit_status"), int)
                        else None
                    ),
                    timestamp=ev.get("timestamp")
                    if isinstance(ev.get("timestamp"), str)
                    else datetime.now().isoformat(),
                )
            )

    return session


def save_session_to_file(
    session: Session,
    session_id: str,
    path: str | Path,
    *,
    include_ctx: bool = True,
) -> dict[str, Any]:
    """Save a session to a JSON file."""
    payload = _session_to_payload(session_id, session, include_ctx=include_ctx)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def load_session_from_file(
    path: str | Path,
    sandbox_config: SandboxConfig,
    context_id: str | None = None,
) -> tuple[str, Session]:
    """Load a session from a JSON file."""
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    resolved_id = context_id or obj.get("context_id") or obj.get("session_id") or "default"
    session = _session_from_payload(obj, resolved_id, sandbox_config)
    return resolved_id, session
