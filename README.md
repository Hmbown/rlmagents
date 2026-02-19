# RLMAgents

[![PyPI - Version](https://img.shields.io/pypi/v/rlmagents)](https://pypi.org/project/rlmagents/)
[![CI](https://github.com/Hmbown/rlmagents/actions/workflows/ci.yml/badge.svg)](https://github.com/Hmbown/rlmagents/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Agent harness for coding and research workflows, built on LangChain + LangGraph.

Fork of LangChain's [Deep Agents](https://github.com/langchain-ai/deepagents) project
with added context isolation, evidence tracking, recursive sub-queries, a restricted
Python REPL, and recipe pipelines. Design follows the
[Recursive Language Model paper](https://arxiv.org/abs/2512.24601).

## How it works

- **Context isolation** -- load large artifacts into named sessions and query them
  without polluting the main conversation (`load_context`, `load_file`).
- **Evidence tracking** -- findings keep provenance metadata so answers can be cited.
- **Recursive sub-queries** -- call `sub_query()` (aliased as `llm_query()`) inside the
  REPL. By default sub-queries use the same model as the main agent unless
  `sub_query_model` is set.
- **Restricted Python REPL** -- run analysis code over loaded contexts. Best-effort
  restriction, not a hard security sandbox.
- **Recipes** -- declarative multi-step pipelines (`validate_recipe`, `estimate_recipe`).
- **`@file` mentions** -- in the CLI, `@path/to/file` in your prompt auto-loads that
  file into an RLM context before the model runs.

## Monorepo layout

| Path | Package | Description |
|------|---------|-------------|
| `libs/rlmagents` | `rlmagents` | Core Python harness |
| `libs/cli` | `rlmagents-cli` | Terminal application |
| `libs/acp` | `deepagents-acp` | Agent Context Protocol integration (e.g. Zed) |
| `libs/harbor` | `deepagents-harbor` | Evaluation framework (Terminal Bench, Harbor) |

## Quickstart

Use the `rlmagents` command as the default way to run the project.
Run from this repository:

```bash
uv run --project libs/cli rlmagents
```

```bash
# One-shot prompt
rlmagents -n "Summarize the repository architecture"

# Resume last conversation
rlmagents -r

# Manage threads and skills
rlmagents threads list
rlmagents skills list
```

The CLI supports interactive and non-interactive modes, conversation threads,
human-in-the-loop approval, remote sandbox backends, and persistent memory
via `.rlmagents/`.

If you need to embed RLMAgents in Python code, use the core API:

```python
from rlmagents import create_rlm_agent

agent = create_rlm_agent()
result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Summarize the key ideas in this paper."}
        ]
    }
)
```

## Paper alignment

The core loop follows Algorithm 1 from arXiv:2512.24601:

1. Context is externalized into REPL sessions (`load_context` / `load_file`).
   CLI `@file` mentions are auto-loaded before model reasoning.
2. The model iterates by writing code, observing execution output, and setting
   a final result.
3. Recursion happens via `sub_query()` / `llm_query()` inside the REPL.

Additional layers beyond the paper: evidence lifecycle and citations,
multi-context search, session persistence (memory-pack JSON), recipe DSL,
context-pressure compaction, and agent-harness middleware.

## Development

Tests and linting are per-package (there is no root-level `make test`):

```bash
# CLI
cd libs/cli && make test
cd libs/cli && make lint

# Core harness
cd libs/rlmagents && uv run --group test pytest tests -q
cd libs/rlmagents && uv run ruff check rlmagents tests

# ACP
cd libs/acp && make test

# Check all lockfiles
make lock-check
```

## Links

- GitHub: https://github.com/Hmbown/rlmagents
- Upstream: https://github.com/langchain-ai/deepagents
