# RLMAgents Integration Setup Guide

**Goal:** Make RLM architecture and workflows innate to the agent by merging the RLM prompt into the core system prompt, adjust configuration to prevent duplication, and ensure `rlmagents-cli` is the one‑package distribution.

**Repository:** `/Users/huntermbown/rlmagents`  
**Current state:** RLM tools work, subagents have full RLM access, tests pass, but the RLM knowledge is appended by middleware rather than being foundational.

## 1. Update the Core System Prompt  
**File:** `libs/cli/deepagents_cli/system_prompt.md`

Insert the following two new sections:

### A. RLM Architecture (after "Core Behavior")  
Add this section immediately after the “Core Behavior” section (after line 18):

```markdown
## RLM (Recursive Language Model) Architecture

You are running in an **RLM‑enhanced** harness that provides structured analysis and evidence‑backed reasoning. This architecture is now fundamental to how you work:

- **Context isolation**: Load large artifacts into named sessions; query them without polluting the main conversation.
- **Evidence tracking**: Keep provenance for findings and generate cited answers.
- **Recursive sub‑queries**: Delegate retrieval/analysis to secondary models with timeouts and budgets.
- **Sandboxed Python REPL**: Run analysis over loaded contexts with 100+ built‑in helpers (regex, extraction, stats, etc.).
- **Recipe pipelines**: Declarative multi‑step workflows for repeatable research and coding tasks.

**Workflow pattern (automatic):**
1. **Load** → `load_context(content, context_id="...")`
2. **Explore** → `search_context`, `peek_context`, `semantic_search`
3. **Analyze** → `exec_python` with helpers (`extract_pattern`, `word_count`, `cite`, etc.)
4. **Reason** → `think` → `evaluate_progress` → `finalize` with evidence
5. **Parallelize** → Spawn subagents (`task` tool) for independent analysis; each subagent has the same full RLM toolset.
```

### B. RLM Tools & Workflows (after "http_request")  
Add this section after the “http_request” subsection (after line 106):

```markdown
### RLM Tools & Workflows

You have 23 RLM tools for structured analysis. Use them instinctively:

**Context management**
- `load_context`, `peek_context`, `search_context`, `semantic_search`, `cross_context_search`
- `save_session`, `load_session`, `diff_contexts`, `list_contexts`

**Analysis & reasoning**
- `exec_python` – sandboxed REPL with helpers: `peek()`, `lines()`, `grep()`, `extract_*()`, `word_count()`, `cite()`, `ctx` variable
- `think`, `evaluate_progress`, `summarize_so_far`, `get_evidence`, `finalize`

**Recipe pipelines**
- `validate_recipe`, `estimate_recipe`, `run_recipe`, `run_recipe_code` – declarative multi‑step workflows

**Subagent orchestration**
- `task(subagent_type="general-purpose", description=...)` – spawns ephemeral subagents with **identical RLM tool access** (same API keys, same REPL). Subagents run in isolated context windows and return only synthesized results.

**Key patterns (internalize these):**
- **Large input?** → `load_context` → analyze in isolation.
- **Parallel independent tasks?** → Spawn multiple `task()` subagents.
- **Structured extraction?** → `exec_python` with `extract_pattern`, `extract_numbers`, etc.
- **Need provenance?** → `cite()` findings, `finalize()` with evidence.
- **Repeatable workflow?** → Build a recipe pipeline.
```

### C. Update “Working with Subagents”  
Amend the existing “Working with Subagents” section (starting at line 126) by adding this bullet point at the end of the list:

```markdown
- **RLM‑enabled subagents**: Every subagent you spawn has full access to the same RLM toolchain (context isolation, Python REPL, evidence tracking, recipes). Use them for parallel analysis, heavy computation, or isolated research.
```

## 2. Disable Duplicate RLM Prompt Injection  
**File:** `libs/cli/deepagents_cli/agent.py`

Modify the call to `create_rlm_agent` (around line 650) to pass an empty `rlm_system_prompt`, preventing the middleware from appending its own prompt:

```python
    agent = create_rlm_agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        backend=composite_backend,
        middleware=agent_middleware,
        interrupt_on=interrupt_on,
        checkpointer=final_checkpointer,
        subagents=custom_subagents or None,
        sub_query_model=sub_query_model,
        rlm_system_prompt="",  # ← Empty string disables duplicate RLM prompt
    ).with_config(config)
```

## 3. Update README with Concrete RLM Examples  
**File:** `README.md` (root of repository)

Insert a new **“RLM Capabilities Examples”** section between the “Example Commands” and “Development” sections (after line 79). Use the following content:

```markdown
## RLM Capabilities Examples

**Context Isolation & Evidence Tracking**

```python
from rlmagents import create_rlm_agent

agent = create_rlm_agent()
# Load large documents into isolated contexts
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "Load this research paper into context 'paper1' and extract key findings with evidence tracking"}
    ]
})
```

**Recursive Sub‑Queries with Python REPL**

```python
# Spawn subagents for parallel analysis with full RLM tool access
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "Analyze these 3 datasets in parallel using subagents with Python REPL, then synthesize results"}
    ]
})
```

**Recipe Pipelines**

```python
# Declarative multi‑step analysis pipelines
recipe = {
    "version": "rlm.recipe.v1",
    "steps": [
        {"op": "load", "content": "Analyze sales data..."},
        {"op": "search", "pattern": "\\d+"},
        {"op": "aggregate", "prompt": "Summarize numeric findings"}
    ]
}
# Recipes can be validated, estimated, and executed
```
```

> **Note:** The code blocks inside the markdown need proper escaping. In the actual edit, ensure the triple backticks are correctly nested (use four backticks for outer code block or escape inner backticks).

## 4. Update rlmagents/AGENTS.md (Optional but Recommended)  
**File:** `libs/rlmagents/AGENTS.md`

Add a section about the integrated RLM architecture:

```markdown
## RLM Architecture Integration

- RLM knowledge is now part of the core system prompt (`system_prompt.md`) rather than being injected by middleware.
- The `RLMMiddleware` still provides the 23 RLM tools, but its system‑prompt injection is disabled in the CLI (`rlm_system_prompt=""`).
- Subagents inherit the same RLM toolset automatically.
- This makes RLM workflows instinctive for the agent — “in your bones” rather than an appended afterthought.
```

## 5. Verify Packaging & Release Readiness  

**A. Test that the changes work**  
Run the existing test suites to ensure nothing is broken:

```bash
cd /Users/huntermbown/rlmagents
make test
cd libs/rlmagents && uv run --group test pytest -q
cd ../cli && uv run --group test pytest tests/unit_tests -q
```

**B. Build both packages**  
```bash
cd libs/rlmagents && uv build
cd ../cli && uv build
```

**C. Release strategy**  
- **Primary package:** `rlmagents-cli` (includes the CLI + SDK via dependency on `rlmagents`)
- **SDK‑only package:** `rlmagents` (for developers who want just the harness)
- **User install:** `pip install rlmagents-cli` gets everything

**D. Update `pyproject.toml` versions**  
Bump version to `0.1.0` in both:
- `libs/rlmagents/pyproject.toml`
- `libs/cli/pyproject.toml`

## 6. Final Checklist  

- [ ] `system_prompt.md` updated with RLM architecture and workflow sections
- [ ] `agent.py` modified to pass `rlm_system_prompt=""`
- [ ] `README.md` enhanced with RLM examples
- [ ] `libs/rlmagents/AGENTS.md` updated (optional)
- [ ] All tests pass
- [ ] Packages build without errors
- [ ] Version numbers bumped to `0.1.0`
- [ ] Ready for PyPI upload (`rlmagents` and `rlmagents-cli`)

---

## Why This Works

1. **“In your bones”**: The RLM knowledge is now part of the core system prompt, not an add‑on.
2. **No duplication**: The middleware’s prompt injection is disabled, preventing redundant text.
3. **Single distribution**: Users install `rlmagents-cli` and get both the CLI and the SDK.
4. **Subagents retain full RLM capabilities** (already true; the prompt now reflects it).
5. **Ready for release**: All tests pass, packaging is configured, and the documentation shows concrete RLM examples.

**Next AI:** Follow the steps above, verify each change, and then proceed to publish the packages to PyPI (or test PyPI). The result is a polished, integrated RLMAgents release where RLM isn’t just an extra feature—it’s the foundation.