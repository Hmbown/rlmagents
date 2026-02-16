"""System prompt loading for the RLM middleware."""

from __future__ import annotations

from pathlib import Path

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "rlm_prompt.md"
_cached_prompt: str | None = None


def load_rlm_prompt() -> str:
    """Load the RLM workflow prompt from rlm_prompt.md."""
    global _cached_prompt
    if _cached_prompt is None:
        _cached_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
    return _cached_prompt
