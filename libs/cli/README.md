# RLMAgents CLI

[![CI](https://github.com/Hmbown/rlmagents/actions/workflows/ci.yml/badge.svg)](https://github.com/Hmbown/rlmagents/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

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
