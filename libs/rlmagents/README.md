# rlmagents

[![PyPI - Version](https://img.shields.io/pypi/v/rlmagents?label=%20)](https://pypi.org/project/rlmagents)
[![PyPI - License](https://img.shields.io/pypi/l/rlmagents)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Agent harness built on LangChain + LangGraph that combines:
- an RLM-aligned core loop (Algorithm 1 from the RLM paper)
- additional agent-harness features (planning, filesystem, sub-agents, recipes, etc.)

Fork of [LangChain Deep Agents](https://github.com/langchain-ai/deepagents).
No runtime dependency on upstream deepagents internals.

Design reference: [Recursive Language Model paper](https://arxiv.org/abs/2512.24601)
([local PDF](../../2512.24601v2.pdf))

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

| Category | Origin | Tools/Behavior |
|----------|--------|----------------|
| Context externalization | RLM core | `load_context`, `load_file_context`, `list_contexts` |
| REPL analysis loop | RLM core | `exec_python`, `get_variable` |
| Recursive decomposition | RLM core | `sub_query()` / `llm_query()` inside REPL execution |
| Completion handoff | RLM-aligned implementation | `finalize` (analogous to paper `Final`) |
| Query and search extensions | RLMAgents extension | `peek_context`, `search_context`, `semantic_search`, `chunk_context`, `cross_context_search`, `rg_search` |
| Planning | RLMAgents extension | `write_todos`, `read_todos` |
| Filesystem | RLMAgents extension | `read_file`, `write_file`, `edit_file`, `ls`, `glob`, `grep` |
| Shell | RLMAgents extension | `execute` (restricted) |
| Sub-agents | RLMAgents extension | `task` (delegate with isolated contexts) |
| Reasoning utilities | RLMAgents extension | `think`, `evaluate_progress`, `summarize_so_far`, `get_evidence` |
| Recipes | RLMAgents extension | `validate_recipe`, `run_recipe`, `run_recipe_code` |
| Memory | RLMAgents extension | AGENTS.md files loaded at startup |
| Skills | RLMAgents extension | Domain-specific capabilities from SKILL.md files |

Tool profiles control which RLM tools are available: `full` (all), `reasoning`
(no recipe/config tools), `core` (minimum set).

## Paper alignment

The RLM-aligned core follows Algorithm 1:

1. Prompt/data are treated as external REPL state instead of root-context stuffing.
2. The model iterates through a code-execute-observe loop.
3. Recursion is programmatic via `sub_query()` / `llm_query()` inside REPL code.

RLMAgents then layers additional functionality not required by the paper:
evidence lifecycle, broader search/retrieval tools, session persistence
(memory-pack JSON), recipe DSL, and full agent-harness middleware
(planning/filesystem/shell/sub-agents/HITL).

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
