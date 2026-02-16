# rlmagents

**RLM-enhanced agents**: Aleph's recursive reasoning with built-in planning, filesystem, and sub-agent capabilities.

[![PyPI - Version](https://img.shields.io/pypi/v/rlmagents?label=%20)](https://pypi.org/project/rlmagents)
[![PyPI - License](https://img.shields.io/pypi/l/rlmagents)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/rlmagents)](https://pypistats.org/packages/rlmagents)

## What is rlmagents?

`rlmagents` is a **standalone** Python package that implements a powerful agent harness with:

- **Built-in planning** - Todo list management for task breakdown
- **Filesystem access** - Read, write, edit, search files
- **Shell execution** - Run commands with sandboxing
- **Sub-agents** - Delegate work with isolated contexts
- **RLM context isolation** - Aleph's recursive reasoning for structured analysis
- **Evidence tracking** - Provenance for all findings
- **Sandboxed Python REPL** - 100+ built-in helpers for analysis
- **Recipe pipelines** - Repeatable multi-step workflows
- **Cross-context search** - Search across multiple isolated contexts
- **Auto-loading** - Large tool results loaded into RLM contexts automatically

Unlike other agent harnesses that require wiring up prompts, tools, and context management yourself, rlmagents provides a complete, ready-to-run agent out of the box.

## Quick Install

```bash
pip install rlmagents
# or
uv add rlmagents
```

## Quick Start

### Basic Usage

```python
from rlmagents import create_rlm_agent

# Create an RLM-enhanced agent
agent = create_rlm_agent()

# Use it like any DeepAgent
result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze this dataset and find patterns..."}]
})
```

### With Skills and Memory

```python
from rlmagents import create_rlm_agent

agent = create_rlm_agent(
    model="claude-sonnet-4-5-20250929",
    skills=["/skills/data-analysis/", "/skills/domain/finance/"],
    memory=["/memory/project-AGENTS.md"],
    auto_load_threshold=5000,  # Auto-load files > 5KB into RLM context
)
```

### With RLM-Enabled Sub-Agents

```python
from rlmagents import create_rlm_agent
from deepagents.middleware.subagents import SubAgent

# Define a specialized sub-agent with RLM capabilities
research_agent: SubAgent = {
    "name": "researcher",
    "description": "Conducts deep research using structured analysis",
    "prompt": "You are a research specialist. Use RLM tools for evidence-backed reasoning.",
    "skills": ["/skills/research/"],
}

# Create main agent with RLM-enabled sub-agents
agent = create_rlm_agent(
    subagents=[research_agent],
    enable_rlm_in_subagents=True,  # Sub-agents get RLM tools automatically
)
```

## Features

### 1. Context Isolation

Load large documents into isolated contexts for efficient analysis:

```python
from rlmagents import create_rlm_agent

agent = create_rlm_agent()

# Agent can use load_context tool:
# 1. Load document into isolated context
# 2. Use search_context, peek_context to explore
# 3. Use exec_python for analysis with 100+ helpers
# 4. Track evidence with citations
```

### 2. Auto-Load Large Results

When DeepAgents tools return large results, they're automatically loaded into RLM contexts:

```python
agent = create_rlm_agent(auto_load_threshold=10_000)  # Default: 10KB

# When read_file returns > 10KB, it's auto-loaded
# Agent gets a hint: "[Large result auto-loaded into RLM context 'auto_read_file']"
# Use search_context/peek_context to explore efficiently
```

### 3. Cross-Context Search

Search across multiple isolated contexts simultaneously:

```python
# Agent can use cross_context_search tool:
# - Search all contexts with one call
# - Results organized by context with source attribution
# - Specify contexts: "context1,context2,context3"
```

### 4. Sandboxed Python REPL

Execute Python code with 100+ built-in helpers for analysis:

```python
# Available helpers inside exec_python:
# Navigation: peek(), lines(), head(), tail()
# Search: search(), grep(), semantic_search(), find_all()
# Extraction: extract_numbers(), extract_emails(), extract_functions()
# Statistics: word_count(), line_count(), word_frequency()
# Text ops: replace_all(), split_by(), chunk()
# Citations: cite() - records evidence with provenance
# Sub-queries: sub_query() - delegate to sub-LLM
```

### 5. Evidence Tracking

All analysis automatically tracks evidence with provenance:

```python
# Use get_evidence tool to review findings
# Each search, extraction, or analysis step is recorded
# Evidence includes source, line range, snippet, and notes
```

### 6. Recipe Pipelines

For repeatable workflows, use recipe pipelines:

```python
# Validate and run declarative recipe pipelines
# Steps: search, chunk, map_sub_query, aggregate, finalize
# Tools: validate_recipe, run_recipe, run_recipe_code
```

## Available RLM Tools (23 Total)

### Context Tools (5)
- `load_context` - Load text into isolated RLM context
- `list_contexts` - List all active contexts with metadata
- `diff_contexts` - Compare two contexts using unified diff
- `save_session` - Serialize context to JSON (memory pack)
- `load_session` - Load previously saved session from JSON

### Query Tools (6)
- `peek_context` - View portion of context by char/line range
- `search_context` - Regex search with line numbers and context
- `semantic_search` - Lightweight semantic search using embeddings
- **`cross_context_search`** - Search across all contexts simultaneously
- `exec_python` - Execute Python code with 100+ built-in helpers
- `get_variable` - Retrieve variable from REPL namespace

### Reasoning Tools (5)
- `think` - Structure a reasoning sub-step
- `evaluate_progress` - Self-evaluate confidence and information gain
- `summarize_so_far` - Generate session summary
- `get_evidence` - Review all collected evidence
- `finalize` - Produce cited final answer

### Recipe Tools (4)
- `validate_recipe` - Validate declarative recipe pipeline
- `estimate_recipe` - Estimate recipe execution cost
- `run_recipe` - Execute recipe pipeline
- `run_recipe_code` - Run recipe code directly

### Status/Meta Tools (2)
- `get_status` - Get RLM system status
- `rlm_tasks` - List active RLM tasks

### Config Tool (1)
- `configure_rlm` - Update runtime configuration

## What's Included Out of the Box

| Feature | Description |
|---------|-------------|
| **Planning** | `write_todos` / `read_todos` for task breakdown and progress tracking |
| **Filesystem** | `read_file`, `write_file`, `edit_file`, `ls`, `glob`, `grep` |
| **Shell access** | `execute` for running commands (with sandboxing) |
| **Sub-agents** | `task` for delegating work with isolated context windows |
| **RLM Context Tools** | `load_context`, `search_context`, `peek_context`, `diff_contexts` |
| **RLM Query Tools** | `semantic_search`, `cross_context_search`, `exec_python`, `get_variable` |
| **RLM Reasoning** | `think`, `evaluate_progress`, `summarize_so_far`, `get_evidence`, `finalize` |
| **RLM Recipes** | `validate_recipe`, `run_recipe`, `run_recipe_code` |
| **Smart defaults** | Prompts that teach the model how to use these tools effectively |
| **Context management** | Auto-summarization when conversations get long |
| **Skills** | Load domain-specific capabilities from SKILL.md files |
| **Memory** | Persistent AGENTS.md context loaded at startup |

## Requirements

- Python 3.11+
- `aleph-rlm>=0.8.5`
- `langchain-core>=1.2.10`
- `langchain>=1.2.10`
- `langchain-anthropic>=0.3.0`
- `langgraph>=0.3.0`
- `pyyaml>=6.0`

### Human-in-the-Loop Approval

```python
agent = create_rlm_agent(
    interrupt_on={
        "edit_file": True,
        "execute": True,
        "run_recipe": True,  # Approve recipe execution
    },
)
```

### Custom RLM System Prompt

```python
custom_prompt = """
# Custom RLM Workflow

You are a data analysis specialist. Always:
1. Load data into RLM context first
2. Use search_context before exec_python
3. Record evidence with cite()
4. Produce cited conclusions
"""

agent = create_rlm_agent(rlm_system_prompt=custom_prompt)
```

### Multiple Isolated Contexts

```python
# Agent can work with multiple contexts simultaneously:
# 1. load_context(content, context_id="source_code")
# 2. load_context(content, context_id="documentation")
# 3. cross_context_search(pattern, contexts="source_code,documentation")
# 4. diff_contexts(a="source_code", b="documentation")
```

### Integration with LangGraph

```python
from rlmagents import create_rlm_agent

agent = create_rlm_agent()

# Use with LangGraph features
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
agent = create_rlm_agent(checkpointer=checkpointer)

# Stream results
for chunk in agent.stream({"messages": [{"role": "user", "content": "..."}]}):
    print(chunk)
```

## Architecture

```
rlmagents/
├── _standalone/          # Incorporated agent harness functionality
│   ├── backends/         # Backend protocol and implementations
│   └── middleware/       # Planning, filesystem, skills, memory, etc.
├── middleware/
│   ├── rlm.py            # RLMMiddleware (23 tools)
│   ├── _tools.py         # RLM tool factories
│   ├── _state.py         # RLMState for LangGraph
│   └── _prompt.py        # RLM system prompt
├── session_manager.py    # Manages Aleph REPL sessions
├── graph.py              # create_rlm_agent() factory
└── base_prompt.md        # Base agent system prompt
```

## Resources

- [Aleph RLM Documentation](https://github.com/anthropics/aleph)
- [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangChain Documentation](https://docs.langchain.com/oss/python)

## Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).

## License

MIT License - see [LICENSE](LICENSE) for details.
