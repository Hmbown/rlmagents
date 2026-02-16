# rlmagents

[![PyPI - Version](https://img.shields.io/pypi/v/rlmagents?label=%20)](https://pypi.org/project/rlmagents)
[![PyPI - License](https://img.shields.io/pypi/l/rlmagents)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**RLM-enhanced agent harness** — A complete, standalone agent framework with planning, filesystem, sub-agents, plus 23 tools for context isolation and evidence-backed reasoning.

Design based on the [Recursive Language Model](https://arxiv.org/abs/2512.24601) (RLM) architecture.

## Quick Start

```bash
pip install rlmagents
# or
uv add rlmagents
```

```python
from rlmagents import create_rlm_agent

agent = create_rlm_agent()
result = agent.invoke({
    "messages": [{"role": "user", "content": "Research this topic and write a summary..."}]
})
```

## What's Included

**Out of the box, rlmagents provides:**

| Feature | Tools |
|---------|-------|
| **Planning** | `write_todos`, `read_todos` |
| **Filesystem** | `read_file`, `write_file`, `edit_file`, `ls`, `glob`, `grep` |
| **Shell** | `execute` (sandboxed) |
| **Sub-agents** | `task` (delegate with isolated contexts) |
| **RLM Context** | `load_context`, `list_contexts`, `diff_contexts`, `save_session`, `load_session` |
| **RLM Query** | `peek_context`, `search_context`, `semantic_search`, `cross_context_search`, `exec_python`, `get_variable` |
| **RLM Reasoning** | `think`, `evaluate_progress`, `summarize_so_far`, `get_evidence`, `finalize` |
| **RLM Recipes** | `validate_recipe`, `run_recipe`, `run_recipe_code` |
| **Memory** | AGENTS.md files loaded at startup |
| **Skills** | Domain-specific capabilities from SKILL.md files |

**Total: 31+ tools** (9 base + 23 RLM + skills/memory)

## RLM Workflow

The agent is taught to use this workflow for complex analysis:

1. **Load** large files/data into isolated RLM contexts
2. **Explore** with `search_context`, `peek_context`, `semantic_search`
3. **Analyze** with `exec_python` (100+ built-in helpers: search, extract, stats, cite)
4. **Track** evidence automatically (provenance for all findings)
5. **Reason** with `think`, `evaluate_progress`
6. **Conclude** with `finalize` (cited answers)

## Configuration

```python
from rlmagents import create_rlm_agent

agent = create_rlm_agent(
    model="claude-sonnet-4-5-20250929",  # Any LangChain model
    skills=["/skills/analysis/"],         # Skill sources
    memory=["/memory/AGENTS.md"],         # Memory files
    auto_load_threshold=5000,             # Auto-load >5KB into RLM
    sandbox_timeout=300.0,                # RLM REPL timeout
    enable_rlm_in_subagents=True,         # RLM tools in sub-agents
    interrupt_on={"edit_file": True},     # Human-in-the-loop
)
```

## Architecture

```
rlmagents/
├── _harness/              # Incorporated agent harness
│   ├── backends/          # Backend protocol (State, Filesystem, etc.)
│   └── middleware/        # Planning, filesystem, skills, memory, etc.
├── middleware/
│   └── rlm.py             # RLM middleware (23 tools)
├── session_manager.py     # Aleph REPL session management
└── graph.py               # create_rlm_agent() factory
```

## Requirements

- Python 3.11+
- `aleph-rlm>=0.8.5` — RLM core (REPL, sessions, evidence)
- `langchain-core>=1.2.10`
- `langchain>=1.2.10`
- `langchain-anthropic>=0.3.0`
- `langgraph>=0.3.0`
- `pyyaml>=6.0`

## Development

```bash
cd libs/rlmagents
uv sync --group test
uv run pytest
uv run ruff check .
uv run ruff format .
```

## Comparison

| Feature | Other Harnesses | rlmagents |
|---------|----------------|-----------|
| Planning | ✅ | ✅ |
| Filesystem | ✅ | ✅ |
| Sub-agents | ✅ | ✅ (RLM-enabled) |
| **Context isolation** | ❌ | ✅ |
| **Evidence tracking** | ❌ | ✅ |
| **Python REPL analysis** | ❌ | ✅ |
| **Cross-context search** | ❌ | ✅ |
| **Auto-load large results** | ❌ | ✅ |
| **Recipe pipelines** | ❌ | ✅ |
| Skills | ✅ | ✅ |
| Memory | ✅ | ✅ |

## License

MIT License — see [LICENSE](LICENSE) for details.
