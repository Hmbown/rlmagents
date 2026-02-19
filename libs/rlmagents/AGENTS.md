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

## RLM-First Operating Model

- Keep large payloads out of live chat history whenever possible.
- For filesystem data, prefer `load_file_context` over inlining file contents.
- Use `search_context`, `peek_context`, `chunk_context`, and `exec_python` against isolated contexts.
- For repo-wide CLI/TUI scans, prefer `rg_search(..., load_context_id=...)` then analyze the loaded context.
- Use `summarize_so_far(clear_history=True)` to compact long reasoning loops.

## Recommended Agent Configuration

```python
from pathlib import Path

from rlmagents import create_rlm_agent

agent = create_rlm_agent(
    model="deepseek/deepseek-chat",
    rlm_tool_profile="core",
    auto_load_threshold=1500,
    auto_load_preview_chars=0,
    rlm_system_prompt=Path("examples/rlm_system_prompt.md").read_text(),
    memory=["examples/AGENTS.md"],
)
```

Profiles:
- `full`: all RLM tools.
- `reasoning`: no recipe/config tools.
- `core`: smallest practical toolset.

## System Prompt Template

Reference template: `examples/rlm_system_prompt.md`.

Required behavior for prompt updates:
- Direct the model to default to isolated RLM contexts for large content.
- Instruct the model to call `load_file_context` for large local files.
- Require evidence-backed conclusions (`get_evidence` + `finalize`).
- Ban large transcript dumps unless explicitly requested.

## AGENTS.md Template

Reference template: `examples/AGENTS.md`.

Template expectations:
- Keep instructions task-specific and short.
- Prefer operational constraints over style-only rules.
- Include explicit context-window limits and offload workflow.

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

- `langchain`, `langgraph` — Agent harness
- no MCP dependency in the package runtime

## RLM Architecture Integration

- RLM workflow knowledge is injected through `rlmagents/rlmagents/rlm_prompt.md`.
- `RLMMiddleware` provides profile-driven RLM tools (`full`, `reasoning`, `core`).
- Subagents inherit the same RLM toolset by default.
- The intended operating model is RLM-native behavior, not an appended optional add-on.

## Releasing

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Commit with conventional commit message
4. Tag and push
