# RLMAgents

[![PyPI - Version](https://img.shields.io/pypi/v/rlmagents)](https://pypi.org/project/rlmagents/)
[![CI](https://github.com/Hmbown/rlmagents/actions/workflows/ci.yml/badge.svg)](https://github.com/Hmbown/rlmagents/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

`rlmagents` is an agent harness built on LangChain + LangGraph.
It uses an RLM-style loop inspired by
[Recursive Language Models](https://arxiv.org/abs/2512.24601)
(Zhang, Kraska, Khattab, 2025), then adds practical agent tooling.

Forked from [LangChain Deep Agents](https://github.com/langchain-ai/deepagents).
Paper reference implementation: [alexzhang13/rlm](https://github.com/alexzhang13/rlm).

## Install

```bash
pip install rlmagents
```

## Usage

### CLI

```bash
rlmagents                        # interactive session
rlmagents -n "Explain this repo" # one-shot
rlmagents -r                     # resume last conversation
```

### Python

```python
from rlmagents import create_rlm_agent

agent = create_rlm_agent()
result = agent.invoke({
    "messages": [{"role": "user", "content": "Summarize the key ideas in this paper."}]
})
```

```python
# With a separate model for recursive sub-calls
agent = create_rlm_agent(
    model="deepseek/deepseek-chat",
    sub_query_model="deepseek/deepseek-reasoner",
    sub_query_timeout=300.0,
    rlm_tool_profile="full",  # full | reasoning | core
)
```

## What Is an RLM?

An RLM (Recursive Language Model) is a way to work on big tasks without
stuffing everything into one huge prompt.

In this project, that means:

- Data can live in REPL/session state instead of only in chat context
- The agent works in a loop by writing and running code (`exec_python`)
- It can call recursive sub-queries with `sub_query` (`llm_query` alias)
- It finishes with `finalize` (or optionally `Final` / `set_final(...)`)

## What It's Helpful For

- Large codebase analysis where you need to inspect many files
- Long documents that are too big for a single prompt
- Multi-step tasks that mix search, code execution, and synthesis
- Workflows that need citations/evidence tracking for final answers
- Repeatable analysis pipelines via recipe tools

## What RLMAgents Adds

- Evidence tracking and citations across tool calls
- Multi-context isolation (separate REPL sessions with `context_id`)
- Recipe tools (`validate_recipe`, `run_recipe`)
- Agent harness tools: planning, filesystem, shell, sub-agents, skills, memory
- `@file` loading plus auto-loading large tool results into contexts

## Monorepo Layout

| Path | Package | Description |
|------|---------|-------------|
| `libs/rlmagents` | `rlmagents` | Python API + bundled CLI/TUI |
| `libs/cli` | `rlmagents-cli` | Standalone CLI package (monorepo development) |
| `libs/acp` | `deepagents-acp` | Agent Context Protocol integration |
| `libs/harbor` | `deepagents-harbor` | Evaluation and benchmark tooling |
| `libs/deepagents` | `deepagents` | Upstream-compatible SDK package |

## Development

```bash
# check all lockfiles in the monorepo
make lock-check

# package-level checks
cd libs/rlmagents
uv sync --group test
uv run pytest tests -q
uv run ruff check rlmagents tests
```

Run from source:

```bash
uv run --project libs/rlmagents rlmagents
```

## Links

- PyPI: https://pypi.org/project/rlmagents/
- GitHub: https://github.com/Hmbown/rlmagents
- Upstream: https://github.com/langchain-ai/deepagents
- Paper: https://arxiv.org/abs/2512.24601
- Paper code: https://github.com/alexzhang13/rlm
