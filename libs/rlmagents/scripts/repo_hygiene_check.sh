#!/bin/bash
# Repo hygiene checks for rlmagents

set -e

ROOT_DIR="$(git rev-parse --show-toplevel)"

echo "Checking for machine-specific path leaks..."
if rg -n "/Volumes/VIXinSSD|/Users/jacoblee" "$ROOT_DIR" -S --glob '!**/uv.lock' --glob '!**/.git/**' --glob '!libs/rlmagents/scripts/repo_hygiene_check.sh'; then
  echo "✗ Found machine-specific paths. Replace with repo-relative or placeholder paths."
  exit 1
fi

echo "Running ruff check..."
uv run ruff check .

echo "Running ruff format check..."
uv run ruff format --check .

echo "Running pytest..."
uv run pytest

echo "✓ All checks passed!"
