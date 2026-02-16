# rlmagents Bootstrap & Verification Prompt

Use this prompt with an AI agent (Claude Code, Cursor, etc.) to verify the rlmagents build, configure it with the provided API keys, and dogfood the repo by using rlmagents itself.

---

## Prompt

```
You are working on the `rlmagents` package at `/Volumes/VIXinSSD/rlmagents/libs/rlmagents/`.

rlmagents is a standalone RLM (Recursive Language Model) agent harness built on LangChain/LangGraph. It provides 23 tools for context isolation, evidence tracking, sandboxed Python REPL, and recursive sub-LLM calls. It was built to match deepagents quality while implementing the RLM architecture from arxiv.org/abs/2512.24601.

## Launch, Telemetry, and Gating Checklist

- One-command launch path:
  - `cd /Volumes/VIXinSSD/rlmagents/libs/rlmagents && uv run python examples/dogfood.py`
- Compatibility check (must return `bootstrap ok`):
  - `cd /Volumes/VIXinSSD/rlmagents/libs/rlmagents && uv run python -c "from examples.bootstrap_config import _load_dotenv_if_available; from rlmagents import create_rlm_agent; _load_dotenv_if_available(); create_rlm_agent(); print('bootstrap ok')"`
- Terminal-flow smoke command:
  - `cd /Volumes/VIXinSSD/rlmagents/libs/rlmagents && uv run pytest tests/unit_tests/test_terminal_bench_scenarios.py -q`
- Optional benchmark score path:
  - `RLMAGENTS_BENCHMARK_SCORE_PATH=$PWD/.artifacts/terminal_bench_score.json uv run pytest tests/unit_tests/test_terminal_bench_scenarios.py -q`

### Success criteria
- **Bootstrap**: compatibility command prints `bootstrap ok`.
- **Model connectivity**: `create_configured_agent()` can build a runnable main model and `sub_query` path when both provider keys are present.
- **Dogfood**: `uv run python examples/dogfood.py` runs and reports tool results or explicit skip messages.
- **Benchmark readiness**: terminal-bench smoke emits all keys:
  - `read_edit_verify_loop`
  - `long_context_compaction`
  - `sub_query_stubbed_path`
  - `dogfood_mocked_provider`

### Failure criteria
- Missing bootstrap script or import errors in `bootstrap_config.py`.
- Missing provider key files/environment variables for any provider-backed path.
- Missing toolchain expectations (`create_rlm_agent`, `RLMSessionManager`, `RLMMiddleware`).
- Terminal-bench smoke fails or emits partial benchmark score.
- `dogfood.py` raises runtime exceptions in `run_tooled_dogfood` or `run_agent`.

## Phase 1: Verify the Build

Run these checks and report any issues:

1. **Tests**: `cd /Volumes/VIXinSSD/rlmagents/libs/rlmagents && uv run pytest tests/ -v`
   - Expect: 70 tests passing

2. **Lint**: `uv run ruff check rlmagents/ tests/`
   - Expect: All checks passed

3. **Import check**: `uv run python -c "from rlmagents import create_rlm_agent, RLMMiddleware, RLMSessionManager; print('OK')"`

4. **No forbidden dependencies**: Verify there are ZERO imports from `aleph`, `aleph-rlm`, or `deepagents` in the main package:
   ```bash
   rg "from aleph|import aleph|from deepagents|import deepagents" rlmagents/ --glob '*.py'
   ```
   - Expect: No results (the _harness/ dir is forked code but should have no aleph/deepagents imports)

5. **Tool count**: Verify 23 RLM tools exist:
   ```python
   from rlmagents.middleware.rlm import _RLM_TOOL_NAMES
   assert len(_RLM_TOOL_NAMES) == 23
   ```

6. **sub_query defaults to None** (not "auto"): Verify `create_rlm_agent` does NOT auto-create any model:
   ```python
   import inspect
   from rlmagents.graph import create_rlm_agent
   sig = inspect.signature(create_rlm_agent)
   assert sig.parameters['sub_query_model'].default is None, "sub_query_model should default to None, not 'auto'"
   assert sig.parameters['model'].default is None, "model should default to None"
   ```

7. **Review key files** for correctness:
   - `rlmagents/graph.py` — create_rlm_agent() factory
   - `rlmagents/middleware/rlm.py` — RLMMiddleware class
   - `rlmagents/session_manager.py` — Session lifecycle + _inject_sub_query
   - `rlmagents/types.py` — Type definitions
   - `rlmagents/repl/sandbox.py` — REPLEnvironment
   - `rlmagents/middleware/_tools.py` — 23 tool builders

Report: What passed, what failed, what needs fixing.

## Phase 2: Configure with API Keys

API keys are stored in `/Volumes/VIXinSSD/rlmagents/libs/rlmagents/.env`:
- `DEEPSEEK_API_KEY` — for main agent model (DeepSeek)
- `MINIMAX_API_KEY` — for recursive sub-query model (MiniMax)

Create a working configuration script at `examples/bootstrap_config.py`:

```python
"""Bootstrap configuration for rlmagents with DeepSeek + MiniMax."""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load API keys
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

from rlmagents import create_rlm_agent

def create_configured_agent(**kwargs):
    """Create an rlmagents agent with DeepSeek (main) + MiniMax (sub_query)."""

    # Main model: DeepSeek
    main_model = init_chat_model(
        "deepseek/deepseek-chat",
        api_key=os.environ["DEEPSEEK_API_KEY"],
    )

    # Sub-query model: MiniMax (for recursive RLM calls in exec_python)
    sub_model = init_chat_model(
        "minimax/minimax-01",
        api_key=os.environ["MINIMAX_API_KEY"],
    )

    return create_rlm_agent(
        model=main_model,
        sub_query_model=sub_model,
        sub_query_timeout=120.0,
        sandbox_timeout=300.0,
        **kwargs,
    )
```

Test that both models respond:
```python
# Quick connectivity test
main_model.invoke("Say hello in one word")
sub_model.invoke("Say hello in one word")
```

Note: You may need to install provider packages:
```bash
uv add python-dotenv
# Check what LangChain provider packages are needed for DeepSeek and MiniMax
# Likely: langchain-openai (DeepSeek is OpenAI-compatible) or langchain-community
```

Figure out the correct `init_chat_model` strings for DeepSeek and MiniMax. DeepSeek uses an OpenAI-compatible API at `https://api.deepseek.com`. MiniMax uses `https://api.minimaxi.chat/v1`. You may need to configure base_url.

## Phase 3: Dogfood — Use rlmagents on Itself

The ultimate test: use rlmagents to analyze its own codebase. Create `examples/dogfood.py`:

1. **Load the rlmagents source into RLM context**:
   ```python
   from rlmagents.session_manager import RLMSessionManager
   from pathlib import Path

   sm = RLMSessionManager()

   # Load key source files
   for name in ["graph.py", "session_manager.py", "middleware/rlm.py", "middleware/_tools.py", "types.py", "repl/sandbox.py"]:
       path = Path("rlmagents") / name
       content = path.read_text()
       sm.create_session(content, context_id=name)

   print(f"Loaded {len(sm.sessions)} contexts")
   ```

2. **Use the tools to analyze**:
   ```python
   from rlmagents.middleware._tools import _build_rlm_tools

   tools = _build_rlm_tools(sm)
   tool_map = {t.name: t for t in tools}

   # Search for patterns
   result = tool_map["search_context"].invoke({"pattern": "sub_query", "context_id": "session_manager.py"})
   print(result)

   # Execute analysis code
   result = tool_map["exec_python"].invoke({
       "code": "print(f'Lines: {line_count()}, Words: {word_count()}')",
       "context_id": "graph.py",
   })
   print(result)

   # Cross-context search
   result = tool_map["cross_context_search"].invoke({"pattern": "def create_"})
   print(result)
   ```

3. **Run the full agent** (requires working API keys):
   ```python
   from examples.bootstrap_config import create_configured_agent

   agent = create_configured_agent()

   result = agent.invoke({
       "messages": [{
           "role": "user",
           "content": (
               "Load the file rlmagents/graph.py into an RLM context, "
               "then analyze it: How many parameters does create_rlm_agent accept? "
               "What middleware does it configure? Give me a structured summary."
           )
       }]
   })

   # Print the agent's response
   for msg in result["messages"]:
       if msg.type == "ai" and msg.content:
           print(msg.content)
   ```

## Phase 4: Report

After completing phases 1-3, provide a summary:

1. **Build status**: Tests, lint, imports — all green?
2. **API connectivity**: Did both DeepSeek and MiniMax respond?
3. **Dogfood results**: Did the agent successfully use its own RLM tools to analyze itself?
4. **Issues found**: Any bugs, missing features, or quality gaps vs deepagents?
5. **Recommendations**: What should be improved before a PyPI release?

Focus especially on:
- Does the sub_query mechanism actually work end-to-end with real models?
- Are the 23 tool descriptions clear enough for an LLM to use them correctly?
- Is the system prompt (base_prompt.md) teaching the RLM workflow effectively?
- How does the agent handle large files (>10KB auto-load threshold)?
```

---

## Key Facts for the Reviewing AI

- **Package location**: `/Volumes/VIXinSSD/rlmagents/libs/rlmagents/`
- **Python**: 3.14 (managed with `uv`)
- **Tests**: 70 passing across 3 test files
- **23 RLM tools**: context(5), query(6), reasoning(5), status(2), recipe(4), config(1)
- **No auto-created models**: Both `model` and `sub_query_model` must be explicitly passed
- **API keys in `.env`** (gitignored): DEEPSEEK_API_KEY, MINIMAX_API_KEY
- **Paper reference**: arxiv.org/abs/2512.24601 (RLM: Recursive Language Models)
- **Algorithm 1 core**: `sub_query()` / `llm_query()` functions injected into REPL namespace
