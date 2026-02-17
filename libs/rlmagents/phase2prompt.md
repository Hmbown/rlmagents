# Phase 2 Execution Prompt: Migrate CLI Module Namespace to `rlmagents_cli`

You are working in:

- Repo root: `/Volumes/VIXinSSD/rlmagents`
- CLI package root: `/Volumes/VIXinSSD/rlmagents/libs/cli`
- Library package root: `/Volumes/VIXinSSD/rlmagents/libs/rlmagents`

## Current Baseline (Already Shipped)

Phase 1 is complete:

- `rlmagents` is the primary CLI command and branding.
- Default harness is `rlmagents`.
- RLMAgents-first config/state paths are in place with deepagents compatibility fallback.
- CLI and library validation is green.

Do **not** regress this behavior while performing namespace migration.

## Objective

Perform a safe Phase 2 module refactor from internal package path `deepagents_cli` to
`rlmagents_cli`, while keeping full backward compatibility for imports, entrypoints,
and legacy script names during transition.

## Non-Negotiable Requirements

1. `rlmagents` remains the primary command and user-facing brand.
2. Canonical implementation package becomes `rlmagents_cli`.
3. `deepagents_cli` remains as a compatibility shim package.
4. `python -m deepagents_cli.main` keeps working via shim delegation.
5. Interactive TUI, non-interactive mode, skills, threads, approvals, and sandbox
   integrations still work.
6. `create_rlm_agent` remains the default runtime harness path.

## Migration Strategy

### 1) Create canonical `rlmagents_cli` package

- Add `libs/cli/rlmagents_cli/` as the canonical package namespace.
- Move/copy implementation modules from `deepagents_cli/` to `rlmagents_cli/`.
- Convert canonical internal imports to `rlmagents_cli.*`.

### 2) Keep `deepagents_cli` as compatibility shim

- Keep `libs/cli/deepagents_cli/` with thin delegating modules.
- Shim behavior:
  - re-export public symbols from `rlmagents_cli`
  - delegate module entrypoints (especially `main.py` and `__main__.py`)
  - optionally emit one-time deprecation warning in entrypoint path only

### 3) Packaging and scripts

- Update `libs/cli/pyproject.toml` scripts to point primary commands to
  `rlmagents_cli:cli_main`.
- Keep legacy aliases (`deepagents`, `deepagents-cli`) mapped through shim or direct
  delegate as transitional compatibility.
- Ensure package data (`*.tcss`, markdown prompts, built-in skills) resolves from
  `rlmagents_cli` canonical path.

### 4) Behavioral invariants

- Keep RLMAgents-first default harness behavior.
- Keep legacy `.deepagents` reads as fallback only.
- Keep all user-facing help/examples/version strings RLMAgents-first.

## Required Validation Commands

Run and report exact output:

1. `cd /Volumes/VIXinSSD/rlmagents/libs/cli && uv run ruff check .`
2. `cd /Volumes/VIXinSSD/rlmagents/libs/cli && uv run pytest tests/unit_tests -q`
3. `cd /Volumes/VIXinSSD/rlmagents/libs/rlmagents && uv run pytest tests/ -q`
4. `cd /Volumes/VIXinSSD/rlmagents/libs/cli && uv run python -m rlmagents_cli.main --help`
5. `cd /Volumes/VIXinSSD/rlmagents/libs/cli && uv run python -m rlmagents_cli.main -n "print hello" --harness rlmagents`
6. Compatibility check: `cd /Volumes/VIXinSSD/rlmagents/libs/cli && uv run python -m deepagents_cli.main --help`

## Static Grep Gates

Run and report:

1. `cd /Volumes/VIXinSSD/rlmagents/libs/cli && rg -n "from deepagents_cli|import deepagents_cli" rlmagents_cli`
   - Expect zero results except explicitly justified compatibility adapters.
2. `cd /Volumes/VIXinSSD/rlmagents/libs/cli && rg -n "deepagents-cli|Deep Agents|Use 'deepagents|~/.deepagents" rlmagents_cli`
   - Expect no canonical UX strings except intentional legacy compatibility notes.
3. `cd /Volumes/VIXinSSD/rlmagents/libs/cli && rg -n "from rlmagents_cli|import rlmagents_cli" deepagents_cli`
   - Expect shim delegates and re-exports.

## Quality Bar

- No circular-import regressions.
- `python -m rlmagents_cli.main` and `python -m deepagents_cli.main` both work.
- Non-interactive path remains functional with RLMAgents harness default.
- Tool approvals, event rendering, and skills loading remain stable.

## Deliverable Format

Return:

1. Migration summary and safety rationale.
2. Files changed grouped by:
   - canonical package (`rlmagents_cli`)
   - shim package (`deepagents_cli`)
   - packaging/scripts
   - tests/docs
3. Exact outputs for all validation and grep gates.
4. Explicit compatibility notes and follow-up plan for eventual shim removal.
