# Project Agent Memory

## RLM Workflow

- Use isolated RLM contexts for all medium and large inputs.
- For large files, call `load_file_context` before analysis.
- Use `search_context`, `peek_context`, and `chunk_context` to inspect targeted regions.
- Use `rg_search(..., load_context_id=...)` for repo-wide scans before deep analysis.
- Use `exec_python` for deterministic extraction and calculations.
- Use `get_evidence` and `finalize` for evidence-backed outputs.

## Context Window Guardrails

- Do not paste long file contents into chat responses.
- Keep tool-output previews short.
- Summarize long reasoning trails with `summarize_so_far(clear_history=True)`.

## Style

- Be direct and concise.
- Report blockers with exact commands, paths, and error messages.
