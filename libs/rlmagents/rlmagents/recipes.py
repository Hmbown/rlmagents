"""Recipe validation and estimation for rlmagents."""

from __future__ import annotations

from typing import Any

RECIPE_SCHEMA_VERSION = "rlm.recipe.v1"

_ALLOWED_OPS: set[str] = {
    "search",
    "peek",
    "lines",
    "take",
    "chunk",
    "filter",
    "map_sub_query",
    "sub_query",
    "aggregate",
    "assign",
    "load",
    "finalize",
}

_ALLOWED_BACKENDS: set[str] = {"auto", "claude", "codex", "gemini", "kimi"}


def _require_positive_int(
    value: Any,
    *,
    field: str,
    errors: list[str],
    min_value: int = 1,
) -> int | None:
    if not isinstance(value, int):
        errors.append(f"{field} must be an integer")
        return None
    if value < min_value:
        errors.append(f"{field} must be >= {min_value}")
        return None
    return value


def _normalize_budget(raw: Any, step_count: int, errors: list[str]) -> dict[str, int]:
    if raw is None:
        return {
            "max_steps": max(step_count, 1),
            "max_sub_queries": 20,
        }

    if not isinstance(raw, dict):
        errors.append("budget must be an object")
        return {
            "max_steps": max(step_count, 1),
            "max_sub_queries": 20,
        }

    max_steps = raw.get("max_steps", step_count)
    max_sub_queries = raw.get("max_sub_queries", 20)

    resolved_steps = _require_positive_int(max_steps, field="budget.max_steps", errors=errors)
    resolved_sub = _require_positive_int(
        max_sub_queries, field="budget.max_sub_queries", errors=errors, min_value=0
    )

    return {
        "max_steps": resolved_steps if resolved_steps is not None else max(step_count, 1),
        "max_sub_queries": resolved_sub if resolved_sub is not None else 20,
    }


def validate_recipe(recipe: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate a recipe payload and return (normalized_recipe, errors)."""
    errors: list[str] = []

    if not isinstance(recipe, dict):
        return None, ["recipe must be an object"]

    version = recipe.get("version", RECIPE_SCHEMA_VERSION)
    if not isinstance(version, str):
        errors.append("version must be a string")
    elif version != RECIPE_SCHEMA_VERSION:
        errors.append(f"unsupported recipe version {version!r}; expected {RECIPE_SCHEMA_VERSION!r}")

    context_id = recipe.get("context_id", "default")
    if not isinstance(context_id, str) or not context_id.strip():
        errors.append("context_id must be a non-empty string")
        context_id = "default"

    raw_steps = recipe.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        errors.append("steps must be a non-empty list")
        return None, errors

    normalized_steps: list[dict[str, Any]] = []

    for idx, raw_step in enumerate(raw_steps):
        path = f"steps[{idx}]"
        if not isinstance(raw_step, dict):
            errors.append(f"{path} must be an object")
            continue

        op = raw_step.get("op")
        if not isinstance(op, str):
            errors.append(f"{path}.op must be a string")
            continue
        if op not in _ALLOWED_OPS:
            errors.append(f"{path}.op '{op}' is not a recognized operation")

        normalized_step: dict[str, Any] = {"op": op}
        for key, value in raw_step.items():
            if key == "op":
                continue
            if key in (
                "context_lines",
                "max_results",
                "start",
                "end",
                "count",
                "chunk_size",
                "overlap",
                "limit",
            ):
                int_val = _require_positive_int(value, field=f"{path}.{key}", errors=errors)
                if int_val is not None:
                    normalized_step[key] = int_val
            elif key == "backend":
                if value not in _ALLOWED_BACKENDS:
                    errors.append(f"{path}.backend '{value}' is not recognized")
                normalized_step[key] = value
            elif key in (
                "pattern",
                "prompt",
                "name",
                "input",
                "store",
                "context_field",
                "contains",
                "field",
            ):
                if isinstance(value, str):
                    normalized_step[key] = value
            elif key == "continue_on_error":
                if isinstance(value, bool):
                    normalized_step[key] = value
            else:
                normalized_step[key] = value

        normalized_steps.append(normalized_step)

    if errors:
        return None, errors

    raw_budget = recipe.get("budget")
    budget = _normalize_budget(raw_budget, len(normalized_steps), errors)

    return {
        "version": version,
        "context_id": context_id,
        "steps": normalized_steps,
        "budget": budget,
    }, []


def estimate_recipe(normalized_recipe: dict[str, Any]) -> dict[str, Any]:
    """Estimate cost and shape for a validated recipe."""
    steps = normalized_recipe.get("steps", [])
    budget = normalized_recipe.get("budget", {})

    step_counts: dict[str, int] = {}
    for step in steps:
        op = step.get("op", "unknown")
        step_counts[op] = step_counts.get(op, 0) + 1

    sub_query_count = step_counts.get("sub_query", 0) + step_counts.get("map_sub_query", 0)

    return {
        "total_steps": len(steps),
        "step_breakdown": step_counts,
        "estimated_sub_queries": sub_query_count,
        "budget": budget,
        "version": normalized_recipe.get("version", RECIPE_SCHEMA_VERSION),
        "context_id": normalized_recipe.get("context_id", "default"),
    }
