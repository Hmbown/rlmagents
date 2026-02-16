# rlmagents

[![PyPI - Version](https://img.shields.io/pypi/v/rlmagents?label=%20)](https://pypi.org/project/rlmagents)
[![PyPI - License](https://img.shields.io/pypi/l/rlmagents)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

RLM-enhanced agent harness: planning, filesystem, sub-agents, plus 23 tools for context isolation and evidence-backed reasoning.

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
    "messages": [{"role": "user", "content": "Analyze this dataset and find patterns..."}]
})
```

## What's Included

**Planning**: Todo list management for task breakdown  
**Filesystem**: Read, write, edit, search files  
**Shell**: Execute commands (sandboxed)  
**Sub-agents**: Delegate work with isolated contexts  
**RLM Tools (23)**: Context isolation, evidence tracking, Python REPL, recipes  

## RLM Workflow

The agent is taught to use this workflow for complex analysis:

1. **Load** context into isolated RLM sessions
2. **Explore** with `search_context`, `peek_context`, `semantic_search`
3. **Analyze** with `exec_python` (100+ built-in helpers)
4. **Track** evidence automatically
5. **Reason** with `think`, `evaluate_progress`
6. **Conclude** with `finalize` (cited answers)

## Available Tools

### Base Agent Tools
- `write_todos`, `read_todos` — Planning
- `read_file`, `write_file`, `edit_file`, `ls`, `glob`, `grep` — Filesystem
- `execute` — Shell commands
- `task` — Sub-agent delegation

### RLM Context Tools
- `load_context` — Load text into isolated context
- `list_contexts` — List active contexts
- `diff_contexts` — Compare two contexts
- `save_session`, `load_session` — Session persistence

### RLM Query Tools
- `peek_context` — View context by range
- `search_context` — Regex search with line numbers
- `semantic_search` — Semantic similarity search
- `cross_context_search` — Search across all contexts
- `exec_python` — Execute Python with helpers
- `get_variable` — Get REPL variable

### RLM Reasoning Tools
- `think` — Structure reasoning sub-steps
- `evaluate_progress` — Assess confidence
- `summarize_so_far` — Session summary
- `get_evidence` — Review collected evidence
- `finalize` — Produce cited conclusion

### RLM Recipe Tools
- `validate_recipe`, `run_recipe`, `run_recipe_code` — Pipeline workflows

## Configuration

```python
agent = create_rlm_agent(
    model="claude-sonnet-4-5-20250929",  # or any LangChain model
    sandbox_timeout=300.0,                 # RLM REPL timeout
    auto_load_threshold=5000,              # Auto-load results >5KB
    context_policy="trusted",              # or "isolated"
    interrupt_on={"edit_file": True},      # Human-in-the-loop
)
```

## Requirements

- Python 3.11+
- `aleph-rlm>=0.8.5`
- `langchain-core>=1.2.10`
- `langchain>=1.2.10`
- `langchain-anthropic>=0.3.0`
- `langgraph>=0.3.0`

## Development

```bash
cd libs/rlmagents
uv sync --group test
uv run pytest
uv run ruff check .
uv run ruff format .
```

## License

MIT License — see [LICENSE](LICENSE) for details.
