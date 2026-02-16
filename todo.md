# next steps

1. run full `rlmagents` validation:
   `cd libs/rlmagents && uv run pytest -q && uv run ruff check .`
2. run bootstrap and dogfood smoke paths end-to-end:
   `cd libs/rlmagents && uv run python examples/bootstrap_config.py`
   `cd libs/rlmagents && uv run python examples/dogfood.py`
3. verify terminal bench flows in CI and inspect published benchmark artifact.
4. confirm provider credentials and model aliases in `.env` for deepseek and minimax.
5. open a PR with an AI-assistance disclaimer and call out high-risk review areas:
   context compaction, execute guardrails, and graph/create_agent compatibility shims.
