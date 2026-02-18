# RLMAgents

`rlmagents` is a batteries-included agent harness for coding and research workflows.
It provides planning, filesystem operations, shell execution, subagents, and context
management on top of LangChain + LangGraph.

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
