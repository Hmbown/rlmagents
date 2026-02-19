# rlmagents

[![PyPI - Version](https://img.shields.io/pypi/v/rlmagents?label=%20)](https://pypi.org/project/rlmagents)
[![PyPI - License](https://img.shields.io/pypi/l/rlmagents)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Agent harness with planning, filesystem, sub-agents, and RLM tools for context
isolation and evidence-tracked reasoning. Built on LangChain + LangGraph.

Fork of [LangChain Deep Agents](https://github.com/langchain-ai/deepagents).
No runtime dependency on upstream deepagents internals.

Design reference: [Recursive Language Model paper](https://arxiv.org/abs/2512.24601)

## Quick start

```bash
pip install rlmagents
# or
uv add rlmagents
```

```bash
# Launch the CLI/TUI
rlmagents
```

```python
from rlmagents import create_rlm_agent

agent = create_rlm_agent()
result = agent.invoke({
    "messages": [{"role": "user", "content": "Research this topic and write a summary..."}]
})
```

## What's included

| Category | Tools |
|----------|-------|
| Planning | `write_todos`, `read_todos` |
| Filesystem | `read_file`, `write_file`, `edit_file`, `ls`, `glob`, `grep` |
| Shell | `execute` (restricted) |
| Sub-agents | `task` (delegate with isolated contexts) |
| Context mgmt | `load_context`, `load_file_context`, `list_contexts`, `diff_contexts`, `save_session`, `load_session` |
| Query | `peek_context`, `search_context`, `semantic_search`, `chunk_context`, `cross_context_search`, `rg_search`, `exec_python`, `get_variable` |
| Reasoning | `think`, `evaluate_progress`, `summarize_so_far`, `get_evidence`, `finalize` |
| Recipes | `validate_recipe`, `run_recipe`, `run_recipe_code` |
| Memory | AGENTS.md files loaded at startup |
| Skills | Domain-specific capabilities from SKILL.md files |

Tool profiles control which RLM tools are available: `full` (all), `reasoning`
(no recipe/config tools), `core` (minimum set).

## Paper alignment

The core loop follows Algorithm 1 from the RLM paper:

1. Context is externalized into REPL sessions (not stuffed into the model window).
2. The model iterates by writing code, observing execution output, and setting a result.
3. Recursion happens via `sub_query()` / `llm_query()` inside the REPL.

Layers beyond the paper: evidence lifecycle, session persistence (memory-pack
JSON), cross-context search, recipe DSL, context-pressure compaction, and
agent-harness middleware (planning, filesystem, sub-agents, HITL).

## Typical workflow

1. **Load** files/data into isolated contexts
2. **Explore** with `search_context`, `peek_context`, `semantic_search`, `rg_search`
3. **Analyze** with `exec_python` (100+ built-in REPL helpers)
4. **Track** evidence (provenance captured automatically)
5. **Reason** with `think`, `evaluate_progress`
6. **Conclude** with `finalize` (cited answers)

## Configuration

```python
from rlmagents import create_rlm_agent

agent = create_rlm_agent(
    model="deepseek/deepseek-chat",       # Main agent model (required)
    sub_query_model="minimax/minimax-01",  # Optional; defaults to reusing `model`
    sub_query_timeout=120.0,
    skills=["/skills/analysis/"],
    memory=["/memory/AGENTS.md"],
    rlm_tool_profile="reasoning",          # full | reasoning | core
    rlm_exclude_tools=("cross_context_search",),
    auto_load_threshold=5000,              # Auto-load tool outputs >5KB into RLM
    auto_load_preview_chars=400,
    sandbox_timeout=300.0,                 # REPL timeout
    interrupt_on={"edit_file": True},      # Human-in-the-loop
)
```

To keep as much as possible outside the chat window:

```python
from pathlib import Path

agent = create_rlm_agent(
    model="deepseek/deepseek-chat",
    rlm_tool_profile="core",
    auto_load_threshold=1500,
    auto_load_preview_chars=0,
    rlm_system_prompt=Path("examples/rlm_system_prompt.md").read_text(),
    memory=["examples/AGENTS.md"],
)
```

## Architecture

```
rlmagents/
├── _harness/              # Agent harness (planning, filesystem, skills, memory)
│   ├── backends/          # Backend protocol (State, Filesystem, etc.)
│   └── middleware/
├── middleware/
│   └── rlm.py             # RLM middleware (tool profiles, auto-load)
├── repl/
│   ├── sandbox.py         # Restricted execution environment
│   └── helpers.py         # Built-in helper functions
├── session_manager.py     # Session lifecycle
├── serialization.py       # Memory-pack serialization
├── recipes.py             # Recipe validation and estimation
└── graph.py               # create_rlm_agent() entry point
```

Security note: the REPL is best-effort restricted by policy and timeouts,
not a formally hardened sandbox.

## Requirements

- Python 3.11+
- `langchain-core>=1.2.10`
- `langchain>=1.2.10`
- `langchain-anthropic>=0.3.0`
- `langgraph>=0.3.0`
- `pyyaml>=6.0`
- `wcmatch>=10.0`

## Development

```bash
cd libs/rlmagents
uv sync --group test
uv run pytest
uv run ruff check .
uv run ruff format .
```

## License

MIT License -- see [LICENSE](LICENSE) for details.
