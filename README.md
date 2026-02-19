# RLMAgents

RLMAgents is an **RLM-native** agent harness for coding and research workflows built on
LangChain + LangGraph.

It extends LangChain's **Deep Agents** project into a release-ready RLM workflow stack:
isolated context sessions, evidence-backed reasoning, recursive sub-queries, a sandboxed
Python analysis REPL, and recipe pipelines.

Upstream Deep Agents project: https://github.com/langchain-ai/deepagents  
RLM design reference: [Recursive Language Model paper](https://arxiv.org/abs/2512.24601)

## RLM architecture (in practice)

- **Context isolation**: load large artifacts into named sessions and query them without
  polluting the main conversation.
- **Evidence tracking**: keep provenance for findings and generate cited answers.
- **Recursive sub-queries**: delegate retrieval/analysis to a secondary model with timeouts
  and budgets.
- **Sandboxed Python REPL**: run analysis over loaded contexts and produce structured outputs.
- **Recipes**: declarative multi-step pipelines for repeatable research and coding tasks.

## What RLMAgents adds (vs Deep Agents)

- First-class RLM toolchain (contexts, evidence, sub-queries, recipes)
- Session serialization (“memory packs”) for resumable work
- `rlmagents-cli`: terminal UI for threads, approvals, skills, and memory
- Opinionated smoke scenarios for harness behavior

## Monorepo Packages

- `libs/rlmagents` — core Python harness (`rlmagents`)
- `libs/cli` — terminal application (`rlmagents-cli`)
- `libs/acp` — Agent Context Protocol support
- `libs/harbor` — evaluation and benchmark tooling

## Quickstart (SDK)

```bash
pip install rlmagents
# or
uv add rlmagents
```

```python
from rlmagents import create_rlm_agent

agent = create_rlm_agent()
result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Research LangGraph and summarize key ideas"}
        ]
    }
)
```

## Quickstart (CLI)

```bash
uv tool install rlmagents-cli
rlmagents
```

The CLI includes:

- Interactive and non-interactive execution
- Conversation resume and thread management
- Human-in-the-loop approval controls
- Remote sandbox integrations
- Persistent memory and skill loading from `.rlmagents`

## Example Commands

```bash
rlmagents -n "Summarize the repository architecture"
rlmagents -r
rlmagents threads list
rlmagents skills list
```

## RLM Capabilities Examples

### Context isolation and evidence tracking

```python
from rlmagents import create_rlm_agent

agent = create_rlm_agent()
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Load this long report into a context named 'report_q1', "
                    "extract the key findings, and return a cited summary."
                ),
            }
        ]
    }
)
```

### Recursive sub-queries and REPL analysis

```python
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Analyze three data sources in parallel with subagents, "
                    "use Python for numeric analysis, then synthesize one answer."
                ),
            }
        ]
    }
)
```

### Recipe pipelines

```python
recipe = {
    "version": "rlm.recipe.v1",
    "steps": [
        {"op": "load", "content": "Quarterly sales data..."},
        {"op": "search", "pattern": "\\d+(?:\\.\\d+)?"},
        {"op": "aggregate", "prompt": "Summarize trends and outliers"},
    ],
}
```

## Development

```bash
# Run monorepo tests
make test

# Lint and format
make lint
make format
```

Package-specific commands:

- `cd libs/cli && uv run --group test pytest tests/unit_tests -q`
- `cd libs/rlmagents && uv run --group test pytest tests -q`
- `cd libs/acp && uv run --group test pytest tests/test_command_allowlist.py -q`

## Repository

- GitHub: `https://github.com/Hmbown/rlmagents`
- CLI source: `libs/cli`
- Harness source: `libs/rlmagents/rlmagents`
