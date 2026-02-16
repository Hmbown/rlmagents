#!/bin/bash
# Repo hygiene checks for rlmagents

set -e

echo "Running ruff check..."
uv run ruff check .

echo "Running ruff format check..."
uv run ruff format --check .

echo "Running pytest..."
uv run pytest

echo "✓ All checks passed!"
