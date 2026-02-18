# Downloading Agents

Agents are just folders. This means you can share, download, and run them instantly.

## Why This Works

- **Agents are folders** — An agent is just an `AGENTS.md` file (memory/instructions) plus a `skills/` directory. No code required.
- **Single artifact** — Package skills and memory together in one zip. Everything the agent needs to run.
- **Run in seconds** — Download, unzip, and run with rlmagents-cli. No setup, no configuration.

## Prerequisites

```bash
uv tool install rlmagents-cli
```

## Quick Start

```bash
# Create a project folder
mkdir my-project && cd my-project && git init

# Download the agent
curl -L https://raw.githubusercontent.com/langchain-ai/deepagents/main/examples/downloading_agents/content-writer.zip -o agent.zip

# Unzip to .rlmagents
unzip agent.zip -d .rlmagents

# Run it
rlmagents
```

## What's Inside

```
.rlmagents/
├── AGENTS.md                    # Agent memory & instructions
└── skills/
    ├── blog-post/SKILL.md       # Blog writing workflow
    └── social-media/SKILL.md    # LinkedIn/Twitter workflow
```

## One-Liner

```bash
git init && curl -L https://raw.githubusercontent.com/langchain-ai/deepagents/main/examples/downloading_agents/content-writer.zip -o agent.zip && unzip agent.zip -d .rlmagents && rm agent.zip && rlmagents
```
