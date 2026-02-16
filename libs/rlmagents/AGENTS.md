# rlmagents Development Guidelines

## Project Structure

```
libs/rlmagents/
├── rlmagents/           # Main package
│   ├── __init__.py
│   ├── graph.py         # create_rlm_agent() factory
│   ├── session_manager.py
│   └── middleware/      # RLM middleware
├── tests/
├── pyproject.toml
└── README.md
```

## Development Commands

```bash
# Install dependencies
uv sync --group test

# Run tests
uv run pytest

# Lint
uv run ruff check .

# Format
uv run ruff format .
```

## Code Style

- Type hints required on all functions
- Google-style docstrings
- American English spelling
- Single backticks for inline code (not double)

## Testing

- Unit tests in `tests/unit_tests/` (no network)
- Integration tests in `tests/integration_tests/` (network OK)
- Use pytest with asyncio mode

## Dependencies

- `aleph-rlm` — RLM core (REPL, sessions, evidence)
- `langchain`, `langgraph` — Agent harness
- NO MCP dependencies

## Releasing

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Commit with conventional commit message
4. Tag and push
