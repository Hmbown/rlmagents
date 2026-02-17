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
from langchain.chat_models import init_chat_model
from rlmagents import create_rlm_agent

# Both main model and sub_query model must be explicitly configured
agent = create_rlm_agent(
    model="deepseek/deepseek-chat",       # Main agent model (required)
    sub_query_model="minimax/minimax-01",  # Recursive sub-LLM (optional)
    sub_query_timeout=120.0,               # Sub-query timeout
    skills=["/skills/analysis/"],          # Skill sources
    memory=["/memory/AGENTS.md"],          # Memory files
    auto_load_threshold=5000,              # Auto-load >5KB into RLM
    sandbox_timeout=300.0,                 # RLM REPL timeout
    enable_rlm_in_subagents=True,          # Deprecated; RLM in sub-agents is always on
    interrupt_on={"edit_file": True},      # Human-in-the-loop
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
├── repl/                  # Sandboxed Python REPL with 100+ helpers
│   ├── sandbox.py         # Sandboxed execution environment
│   └── helpers.py         # Built-in helper functions
├── session_manager.py     # Session lifecycle management
├── serialization.py       # Session serialization (memory packs)
├── recipes.py             # Recipe validation and execution
└── graph.py               # create_rlm_agent() factory
```

## Requirements

- Python 3.11+
- `langchain-core>=1.2.10`
- `langchain>=1.2.10`
- `langchain-anthropic>=0.3.0`
- `langgraph>=0.3.0`
- `pyyaml>=6.0`
- `wcmatch>=10.0`

**No external dependencies on aleph-rlm or deepagents** — rlmagents is fully standalone.

## Development

```bash
cd libs/rlmagents
uv sync --group test
uv run pytest
uv run ruff check .
uv run ruff format .
```

## Launch & Telemetry Checklist

- One-command launch path:
  ```bash
  (cd "$(git rev-parse --show-toplevel)/libs/rlmagents" && uv run python examples/dogfood.py)
  ```

- Compatibility check command:
  ```bash
  (cd "$(git rev-parse --show-toplevel)/libs/rlmagents" && \
  uv run python -c "from examples.bootstrap_config import _load_dotenv_if_available; from rlmagents import create_rlm_agent; _load_dotenv_if_available(); create_rlm_agent(); print('bootstrap ok')")
  ```

- Terminal flow smoke command:
  ```bash
  (cd "$(git rev-parse --show-toplevel)/libs/rlmagents" && \
  uv run pytest tests/unit_tests/test_terminal_bench_scenarios.py -q)
  ```

- Benchmark-score output (optional when benchmark job is running):
  ```bash
  (cd "$(git rev-parse --show-toplevel)/libs/rlmagents" && \
  RLMAGENTS_BENCHMARK_SCORE_PATH=$PWD/.artifacts/terminal_bench_score.json \
    uv run pytest tests/unit_tests/test_terminal_bench_scenarios.py -q)
  ```

Expected JSON output format when score output is enabled:

```json
{
  "read_edit_verify_loop": "passed",
  "long_context_compaction": "passed",
  "sub_query_stubbed_path": "passed",
  "dogfood_mocked_provider": "passed"
}
```

Model- and run-time telemetry checks should include:

- Whether `create_configured_agent()` can initialize (or skip with explicit env-based reason).
- Whether terminal-bench scenarios report all keys above.
- Whether `examples/dogfood.py` executes `run_tooled_dogfood()` and prints agent output when keys are present.

### Success criteria

- **Bootstrap**: `examples/bootstrap_config.py` imports, and `uv run python -c ...` check returns
  `bootstrap ok`.
- **Model connectivity**: `create_configured_agent()` returns a runnable agent object when provider keys
  are present, and `create_rlm_agent` is constructible with mocked tool call flows.
- **Dogfood readiness**: `uv run python examples/dogfood.py` executes `run_tooled_dogfood()` and, when
  keys are present, prints model output without uncaught exceptions.
- **Terminal-flow readiness**: scenario smoke tests in
  `tests/unit_tests/test_terminal_bench_scenarios.py` pass.
- **Benchmark readiness**: scenario smoke tests emit the terminal bench score artifact (when enabled)
  and include all four scenario keys.

### Failure criteria

- Any syntax/import error in `rlmagents` or `examples/bootstrap_config.py`.
- Missing environment variables for provider-backed flows (`DEEPSEEK_API_KEY` or `MINIMAX_API_KEY`).
- `uv run pytest tests/unit_tests/test_terminal_bench_scenarios.py` failures.
- `create_configured_agent()` or `dogfood.py` raising runtime exceptions instead of exiting with explicit
  skip/failure output.

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
