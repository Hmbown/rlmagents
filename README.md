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

## Paper Alignment (What We Claim)

This project is paper-aligned at the loop level, but it is not the official
paper codebase.

The core behavior we share with the paper:

- Prompt/data can live outside the root model context (in REPL/session state)
- The model iterates by writing and executing code (`exec_python`)
- Recursive sub-calls are programmatic via `sub_query` (`llm_query` alias)
- Completion supports explicit `finalize` and optional `Final` / `set_final(...)`

## What RLMAgents Adds

- Evidence tracking and citations across tool calls
- Multi-context isolation (separate REPL sessions with `context_id`)
- Recipe tools (`validate_recipe`, `run_recipe`)
- Agent harness tools: planning, filesystem, shell, sub-agents, skills, memory
- `@file` auto-loading and auto-loading large tool results into contexts

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
