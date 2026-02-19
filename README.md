# RLMAgents

[![PyPI - Version](https://img.shields.io/pypi/v/rlmagents)](https://pypi.org/project/rlmagents/)
[![CI](https://github.com/Hmbown/rlmagents/actions/workflows/ci.yml/badge.svg)](https://github.com/Hmbown/rlmagents/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Agent harness for coding and research workflows, built on LangChain + LangGraph.

Fork of LangChain's [Deep Agents](https://github.com/langchain-ai/deepagents) project.
This repo includes both:
- an RLM-aligned core loop based on Algorithm 1 in the
  [Recursive Language Model paper](https://arxiv.org/abs/2512.24601)
  ([local PDF](2512.24601v2.pdf))
- additional RLMAgents features layered on top of that core

As of `rlmagents==0.0.3`, a single install provides both:
- the Python API (`from rlmagents import create_rlm_agent`)
- the terminal CLI/TUI (`rlmagents`)

## RLM core vs additions

### RLM core (paper-aligned)

- **Prompt externalization** -- prompts/data live in REPL-backed contexts
  (`load_context`, `load_file_context`) instead of being stuffed into a single
  model call.
- **Code-execute-observe loop** -- the model iterates by writing code and observing
  execution output (`exec_python`).
- **Symbolic recursive calls** -- `sub_query()` / `llm_query()` can be invoked from
  inside REPL code for recursive decomposition.
- **Explicit completion step** -- responses are concluded with `finalize` (analogous
  to setting `Final` in Algorithm 1).

### RLMAgents additions (beyond the paper)

- **Evidence tracking and citations** -- provenance is recorded across tool calls.
- **Multi-context lifecycle tools** -- diff/search/save/load workflows across contexts.
- **Recipe DSL** -- declarative, multi-step pipelines (`validate_recipe`,
  `estimate_recipe`, `run_recipe`).
- **Agent harness tools** -- planning, filesystem, shell, and sub-agent delegation.
- **CLI ergonomics** -- `@file` mentions auto-load files into contexts before reasoning.

## Monorepo layout

| Path | Package | Description |
|------|---------|-------------|
| `libs/rlmagents` | `rlmagents` | Published package (Python API + bundled CLI/TUI) |
| `libs/cli` | `rlmagents-cli` | Standalone CLI package kept for monorepo development |
| `libs/acp` | `deepagents-acp` | Agent Context Protocol integration (e.g. Zed) |
| `libs/harbor` | `deepagents-harbor` | Evaluation framework (Terminal Bench, Harbor) |

## Quickstart

Install from PyPI:

```bash
pip install rlmagents
```

Then run the CLI:

```bash
rlmagents

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

### From source

If you're developing locally, you can run directly from the repo with
[uv](https://docs.astral.sh/uv/):

```bash
# Run from the bundled package (matches published behavior)
uv run --project libs/rlmagents rlmagents

# Or run from the standalone CLI package while developing CLI-only changes
uv run --project libs/cli rlmagents
```

### Python API

To embed RLMAgents in Python code, use the core API:

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

## Paper alignment notes

The implementation follows Algorithm 1's core shape (paper-inspired, not a
line-by-line reproduction):

1. `InitREPL(prompt=P)` equivalent via context loading into REPL sessions.
2. Iterative `code -> execute -> observe` behavior in the REPL loop.
3. Recursive sub-calls through `sub_query()` / `llm_query()` from REPL code.

It also intentionally avoids the paper's Algorithm 2 pitfalls:

1. Long input is handled as external context, not prepended wholesale to root history.
2. Sub-calls are programmatic (inside code), not only verbalized one-off delegations.
3. Intermediate results are stored in REPL/session state, not only transient chat text.

Everything listed in `RLM core vs additions -> RLMAgents additions` is implementation
scope added by this project.

## Paper-parity TODOs

These are the main areas where the current implementation is still partial
relative to the paper's idealized scaffold:

- [ ] Add an explicit, inspectable root-loop history artifact (`hist`) that
      captures executed code plus bounded execution metadata across iterations.
- [ ] Auto-inject constant-size per-iteration execution metadata into root model
      context by default, instead of relying only on explicit tool calls.
- [ ] Add an optional built-in finish sentinel (paper `Final` style) so loop
      termination can be enforced without requiring an explicit `finalize` call.

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
