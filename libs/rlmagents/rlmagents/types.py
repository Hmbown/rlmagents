"""Shared type definitions for rlmagents."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal

# -----------------------------------------------------------------------------
# Context Types
# -----------------------------------------------------------------------------


class ContentFormat(Enum):
    """Detected or specified format of context data."""

    TEXT = "text"
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    CODE = "code"
    BINARY = "binary"
    MIXED = "mixed"


@dataclass(slots=True)
class ContextMetadata:
    """Metadata about the loaded context."""

    format: ContentFormat
    size_bytes: int
    size_chars: int
    size_lines: int
    size_tokens_estimate: int
    structure_hint: str | None
    sample_preview: str


# A single context payload can be text, bytes, or JSON-like.
ContextType = str | bytes | dict | list | tuple


# -----------------------------------------------------------------------------
# Execution Types
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class ExecutionResult:
    """Result of executing code in the sandbox REPL."""

    stdout: str
    stderr: str
    return_value: object | None
    variables_updated: list[str]
    truncated: bool
    execution_time_ms: float
    error: str | None


# -----------------------------------------------------------------------------
# Evidence and Session Types
# -----------------------------------------------------------------------------

EvidenceSource = Literal[
    "search",
    "peek",
    "exec",
    "manual",
    "action",
    "sub_query",
    "cross_context_search",
]
_VALID_EVIDENCE_SOURCES: set[str] = {
    "search",
    "peek",
    "exec",
    "manual",
    "action",
    "sub_query",
    "cross_context_search",
}


@dataclass
class Evidence:
    """Provenance tracking for reasoning conclusions."""

    source: EvidenceSource
    snippet: str
    source_op: str | None = None
    context_id: str | None = None
    file_path: str | None = None
    line_range: tuple[int, int] | None = None
    pattern: str | None = None
    note: str | None = None
    command_exit_status: int | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())  # noqa: DTZ005


# -----------------------------------------------------------------------------
# Sub-query Types
# -----------------------------------------------------------------------------

SubQueryFn = Callable[[str, str | None], Awaitable[str]]
