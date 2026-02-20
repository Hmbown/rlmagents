# RLMAgents

[![PyPI - Version](https://img.shields.io/pypi/v/rlmagents)](https://pypi.org/project/rlmagents/)
[![CI](https://github.com/Hmbown/rlmagents/actions/workflows/ci.yml/badge.svg)](https://github.com/Hmbown/rlmagents/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Agent harness built on LangChain + LangGraph. Based on
[Recursive Language Models](https://arxiv.org/abs/2512.24601)
(Zhang, Kraska, Khattab 2025) with additional agent tooling.

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

## What's in it

### RLM core

The paper's key insight is that long prompts should not be fed directly into
the model. Instead, the prompt is treated as part of an external environment
that the model interacts with symbolically and recursively.

In practice this means three things (corresponding to the three design choices
that distinguish Algorithm 1 from Algorithm 2 in the paper):

1. **Prompt is external.** The prompt is loaded as a variable in a Python REPL
   environment. The model receives only constant-size metadata (length, short
   prefix). This is what the paper calls giving the model a "symbolic handle"
   to the prompt.

2. **Recursion is programmatic.** The model can invoke a sub-LLM from inside
   REPL code (the paper uses `llm_query`; we expose this as `sub_query()` with
   `llm_query()` as an alias). This lets the model write loops that call the
   sub-LLM on slices of the prompt, rather than being limited to verbalized
   one-off delegations.

3. **Intermediate results live in REPL state.** Variables and sub-call outputs
   are stored in the REPL, not in the model's context window. Only compact
   metadata of each execution step is appended to the root history (`hist`),
   which is compacted when it exceeds the model's context budget.

Iteration stops when the model sets `Final` in the REPL (we also support
`set_final(...)` and an explicit `finalize` tool call).

The paper's experiments used max recursion depth 1 for sub-calls (sub-calls
are flat LM invocations, not full RLM loops). Our default matches this.

### Additions (not in the paper)

- Evidence tracking and citations across tool calls
- Multi-context isolation (work with multiple data sources in separate REPL sessions)
- `sub_query_map` (parallel fan-out over multiple prompts), `sub_query_strict` (output validation)
- Recipe DSL for declarative multi-step pipelines (`validate_recipe`, `run_recipe`)
- Agent harness: planning, filesystem, shell, sub-agents, skills, memory
- `@file` auto-loading and auto-loading large tool results into contexts

### `sub_query` usage

Inside `exec_python`, use `sub_query` when you need LLM judgment on data
(not just regex or Python):

```python
# Fan out over chunks
chunks = chunk(2000, 200)
summaries = sub_query_map([f"Summarize:\n{c}" for c in chunks])

# Classification
verdict = sub_query("Is this code safe?", code_block)

# Multi-hop
entities = sub_query("Extract named entities", ctx)
rels = sub_query(f"Relationships between: {entities}", ctx)
```

For tasks needing filesystem access or true parallelism, use sub-agents
(`task` tool) instead.

## Monorepo layout

| Path | Package | Description |
|------|---------|-------------|
| `libs/rlmagents` | `rlmagents` | Python API + bundled CLI/TUI |
| `libs/cli` | `rlmagents-cli` | Standalone CLI (monorepo dev) |
| `libs/acp` | `deepagents-acp` | Agent Context Protocol (Zed) |
| `libs/harbor` | `deepagents-harbor` | Evaluation (Terminal Bench) |

## Development

```bash
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
