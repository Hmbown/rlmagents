# RLMAgents CLI

[![PyPI - Version](https://img.shields.io/pypi/v/rlmagents-cli?label=%20)](https://pypi.org/project/rlmagents-cli/)
[![PyPI - License](https://img.shields.io/pypi/l/rlmagents-cli)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pepy/dt/rlmagents-cli)](https://pypistats.org/packages/rlmagents-cli)

`rlmagents-cli` is the terminal interface for `rlmagents`: interactive agent execution with
planning, filesystem tools, shell execution, sub-agents, and the full RLM context toolchain.

It extends the Deep Agents CLI lineage with an RLM-native workflow model grounded in the
[Recursive Language Model paper](https://arxiv.org/abs/2512.24601).

## Quick Install

```bash
uv tool install rlmagents-cli
rlmagents
```

## What You Get

- Interactive + non-interactive runs
- Human-in-the-loop approvals
- RLM context isolation and evidence-backed reasoning
- Recursive sub-query workflows with REPL-assisted analysis
- Skills and memory loading from `.rlmagents`
- CLI defaults tuned for context-window efficiency (`reasoning` RLM tool profile)
- Optional repo-wide scan workflow via `rg_search(..., load_context_id=...)` when broad discovery is needed

## Resources

- [Repository](https://github.com/Hmbown/rlmagents)
- [Deep Agents Upstream](https://github.com/langchain-ai/deepagents)
- [CLI Source](https://github.com/Hmbown/rlmagents/tree/main/libs/cli)
- [RLM Harness Source](https://github.com/Hmbown/rlmagents/tree/main/libs/rlmagents)
- [Issue Tracker](https://github.com/Hmbown/rlmagents/issues)
