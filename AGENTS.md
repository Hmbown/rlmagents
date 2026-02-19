# AGENTS.md

This guide is for both human contributors and coding agents working in `Hmbown/rlmagents`.
Use it as the repo-level source of truth for implementation, testing, and pull requests.

## Goals

- Keep behavior stable for existing users.
- Keep changes small, reviewable, and well-tested.
- Keep documentation and code aligned.

## Repository Map

This is a Python monorepo with independently versioned packages.

| Path | Package | Purpose |
| --- | --- | --- |
| `libs/rlmagents` | `rlmagents` | Core RLM harness |
| `libs/cli` | `rlmagents-cli` | Terminal app and agent UX |
| `libs/acp` | `deepagents-acp` | Agent Context Protocol integration |
| `libs/harbor` | `deepagents-harbor` | Evaluation and benchmark tooling |
| `libs/deepagents` | `deepagents` | Upstream-compatible SDK package |
| `.github/workflows` | n/a | CI/CD and release automation |

## Toolchain

- `uv` for dependency management and lockfiles.
- `make` for common tasks.
- `ruff` for lint/format checks.
- `ty` for type-checking where configured.
- `pytest` for tests.

Each package in `libs/` has its own `pyproject.toml` and `uv.lock`.

## Fast Start

Use package-local commands instead of assuming a root-level `make test`.

```bash
# lockfile consistency for the monorepo
make lock-check

# refresh lockfiles when needed
make lock
```

```bash
# CLI package
cd libs/cli && make lint
cd libs/cli && make test

# Core harness
cd libs/rlmagents && uv run ruff check rlmagents tests
cd libs/rlmagents && uv run --group test pytest tests -q

# ACP package
cd libs/acp && make test
```

## Public API Stability (Critical)

Avoid breaking changes in exported interfaces.

Before changing a public function or class:

- Check whether it is exported in `__init__.py`.
- Check usage in tests, examples, and docs.
- Prefer additive changes with keyword-only params:
  `def f(a: str, *, new_flag: bool = False) -> None: ...`
- Call out behavior changes explicitly in docs and PR notes.

Rule of thumb: if user code from last week can break, redesign the change.

## Code Standards

- Add full type hints on function parameters and return values.
- Avoid `Any` unless there is no practical alternative.
- Prefer clear, descriptive names over short abbreviations.
- Keep functions focused; split only when it improves readability.
- Remove dead code and stale comments before commit.

## Ruff Suppression Policy

Use inline suppression for one-off exceptions:

```python
timeout = 30  # noqa: PLR2004  # default HTTP timeout value
```

Use `[tool.ruff.lint.per-file-ignores]` only for categorical policy (for example test-only conventions), not for individual lines.

## Testing Standards

Every bug fix and feature should have test coverage.

- Unit tests belong in `tests/unit_tests/` and must not require network access.
- Integration tests belong in `tests/integration_tests/` and may use network calls.
- Mirror source layout in test layout where practical.
- Prefer behavior-focused tests over implementation-detail tests.
- Keep tests deterministic and avoid flaky timing assumptions.

## Security and Reliability

- Never use `eval()`, `exec()`, or `pickle` on untrusted input.
- Do not use bare `except:`; catch specific exceptions.
- Ensure files, sockets, and subprocesses are cleaned up.
- Validate and sanitize user-controlled inputs.

## Documentation Standards

Use Google-style docstrings for public functions.

```python
def send_email(to: str, msg: str, *, priority: str = "normal") -> bool:
    """Send an email message.

    Args:
        to: Recipient email address.
        msg: Message body.
        priority: Delivery priority label.

    Returns:
        `True` when delivery succeeds, else `False`.
    """
```

- Keep types in signatures, not in docstring fields.
- Explain intent and constraints, not only mechanics.
- Use American English spelling.
- Use single-backtick inline code formatting.

## CLI Package Notes (`libs/cli`)

The CLI uses [Textual](https://textual.textualize.io/).

- Prefer `@work` workers for async tasks.
- Use message passing for widget coordination.
- Use reactive attributes for stateful UI behavior.
- Use `textual.pilot` for async UI tests.

References:

- [Textual Guide](https://textual.textualize.io/guide/)
- [Testing Guide](https://textual.textualize.io/guide/testing/)
- [Widget Gallery](https://textual.textualize.io/widget_gallery/)
- [CSS Reference](https://textual.textualize.io/styles/)

## Commits and Pull Requests

Commit and PR titles must follow Conventional Commits and include a lowercase scope.

Examples:

```txt
feat(cli): add thread export command
fix(rlmagents): handle empty evidence results
chore(harbor): refresh benchmark fixtures
docs(readme): simplify quickstart flow
```

PR requirements:

- Explain why the change is needed.
- Highlight risky or high-review areas.
- Include test evidence or rationale when tests are not added.
- Add a short AI-assistance disclosure when AI tools were used.

Use `.github/workflows/pr_lint.yml` as the enforcement source for title rules.

## External References

- Project docs: https://docs.langchain.com/oss/python/deepagents/overview
- Upstream docs repo: https://github.com/langchain-ai/docs
- Upstream contributing guide: https://docs.langchain.com/oss/python/contributing/overview
