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

## Paper Alignment and RLMAgents Extensions

RLMAgents follows the core RLM pattern from the paper (arXiv:2512.24601):

- Prompt content is stored in an external REPL variable and accessed symbolically.
- The model iterates by writing code, observing execution feedback, and setting a final result.
- Recursion is programmatic through `sub_query()`/`llm_query()` inside the REPL.

RLMAgents also adds implementation layers that are not part of the paper's core algorithm:

- Evidence and citation lifecycle (`get_evidence`, provenance metadata, pruning policy)
- Multi-context operations (`cross_context_search`, context diffing across sessions)
- Session persistence (`save_session`/`load_session`, memory-pack JSON schema)
- Recipe system (`validate_recipe`, `estimate_recipe`, `run_recipe`, DSL compilation)
- Context-pressure policy (including rlmagents-specific compaction heuristics)
- Agent-harness integrations (auto-loading large tool outputs, sub-agent-wide RLM middleware)

If one of these extension features fails, treat it as an RLMAgents implementation issue,
not user error and not a failure of the base RLM paper design.

## What RLMAgents adds (vs Deep Agents)

- First-class RLM toolchain (contexts, evidence, sub-queries, recipes)
- Session serialization (“memory packs”) for resumable work
- `rlmagents-cli`: terminal UI for threads, approvals, skills, and memory
- Opinionated smoke scenarios for harness behavior

## Practical Value Assessment

After comprehensive testing, the RLM features provide genuine value for research and analysis workflows:

### ✅ Context Isolation
- **Tested**: Loaded multiple documents into separate contexts, performed cross-context searches
- **Value**: Enables parallel analysis of large documents without polluting conversation context
- **Use case**: Research papers, codebases, logs analyzed side-by-side

### ✅ Evidence Tracking & Provenance
- **Tested**: All searches, Python executions, and manual citations automatically tracked
- **Value**: Provides verifiable audit trail for findings; essential for research documentation
- **Use case**: Academic research, legal analysis, audit compliance

### ✅ Python REPL with 100+ Helpers
- **Tested**: Used built-in functions (search, extract_numbers, word_count, cite) for data analysis
- **Value**: Enables complex data manipulation beyond LLM capabilities; sandboxed for safety
- **Use case**: Data extraction, statistical analysis, text processing, CSV/JSON parsing

### ✅ Structured Reasoning Workflow
- **Tested**: Used think → evaluate → finalize workflow for systematic analysis
- **Value**: Reduces ad-hoc prompting; provides confidence tracking and progress assessment
- **Use case**: Complex problem-solving, research synthesis, decision support

### ✅ Recipe Pipelines
- **Examined**: Declarative multi-step workflows with budget controls
- **Value**: Enables repeatable analysis pipelines; manages token usage and costs
- **Use case**: Batch processing, ETL workflows, automated reporting

**Assessment summary**: RLM features are helpful for research, data analysis, and complex reasoning tasks. In this implementation, they provide a systematic and verifiable workflow with different tradeoffs than baseline agent loops.

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
