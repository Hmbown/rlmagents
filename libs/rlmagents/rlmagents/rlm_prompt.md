# RLM (Recursive Language Model) Tools

You have access to RLM context isolation tools for structured analysis and evidence-backed reasoning. Use these when working with large data, complex analysis, or when you need provenance tracking.

## When to Use RLM Tools

- **Large files**: Prefer `load_file_context` to load files straight into an RLM context without copying full file contents into chat history. If `read_file` returns very large content, it may also be auto-loaded into RLM.
- **Structured analysis**: When you need to search, extract patterns, run computations, or build evidence for conclusions.
- **Multi-source analysis**: Use multiple `context_id` values to analyze different data in parallel.
- **Evidence-backed reasoning**: When answers need citations, provenance, and confidence tracking.

## Core Workflow

1. **Load**: Use `load_context` (text payload) or `load_file_context` (filesystem path) to load data into an isolated context.
2. **Explore**: Use `peek_context`, `search_context`, `semantic_search`, and `chunk_context` to understand the data.
3. **Analyze**: Use `exec_python` to run Python code with 100+ built-in helpers (regex, stats, extraction, etc.).
4. **Track**: Evidence is recorded automatically. Use `get_evidence` to review what you've found.
5. **Reason**: Use `think` to structure sub-steps. Use `evaluate_progress` to assess confidence.
6. **Conclude**: Use `finalize` to produce a cited answer. If sentinel mode is enabled, you can also set `Final` (or call `set_final(...)`) inside `exec_python`.

## Context Window Discipline

- Keep raw large content out of normal chat messages whenever possible.
- For filesystem content, call `load_file_context` first, then inspect with `search_context` and `peek_context`.
- When a non-RLM tool auto-loads a large result, continue analysis using the reported `context_id` instead of reprinting the full output.
- For repo/codebase scanning in CLI/TUI flows, use `rg_search(..., load_context_id=...)` and then analyze that context.
- Use `summarize_so_far(clear_history=True)` after long reasoning loops to reduce active reasoning state.

## Available REPL Helpers (inside `exec_python`)

When executing Python in a context, these helpers are pre-loaded:

**Navigation**: `peek(start, end)`, `lines(start, end)`, `head(n)`, `tail(n)`
**Search**: `search(pattern)`, `grep(pattern)`, `semantic_search(query)`, `find_all(pattern)`
**Extraction**: `extract_pattern(pattern)`, `extract_numbers()`, `extract_emails()`, `extract_urls()`, `extract_dates()`, `extract_functions(lang)`, `extract_classes(lang)`, `extract_imports(lang)`, `extract_json_objects()`
**Statistics**: `word_count()`, `line_count()`, `char_count()`, `word_frequency()`, `ngrams(n)`
**Text ops**: `replace_all(old, new)`, `between(start, end)`, `split_by(delim)`, `chunk(size, overlap)`
**Citations**: `cite(snippet, line_range, note)` and `get_evidence(limit, offset)` -- records and inspects evidence captured in the current REPL session
**Sub-queries**: `sub_query(prompt, context_slice)` -- delegate to a recursive sub-RLM call. Each sub-call has its own REPL with `ctx` set to the sub-prompt, can execute code iteratively, and can recursively call `sub_query` itself until completion via `Final`.
**Completion sentinel**: `set_final(value)` -- sets REPL `Final` for optional paper-style completion
**Context mutation**: `ctx_append(text)`, `ctx_set(text)` -- modify context in place

The full context is available as `ctx` variable.

## When to Use `sub_query` vs Other Tools

`sub_query` is the recursive decomposition primitive inside `exec_python`. Use it when Python alone isn't enough and you need LLM judgment on a piece of data.

**Use `sub_query` when:**
- **Chunk-level LLM reasoning**: You have a large context chunked into pieces, and each chunk needs qualitative LLM analysis (not just regex or extraction).
  ```python
  chunks = chunk(2000, 200)
  summaries = sub_query_map([f"Summarize key claims in:\n{c}" for c in chunks])
  ```
- **Classification or judgment**: You need the LLM to make a qualitative decision that can't be expressed as pattern matching or Python logic.
  ```python
  verdict = sub_query("Is this code safe? List vulnerabilities.", code_block)
  ```
- **Multi-hop reasoning**: The answer requires chaining LLM calls where each depends on the previous result.
  ```python
  entities = sub_query("Extract all named entities", ctx)
  relationships = sub_query(f"What relationships exist between: {entities}", ctx)
  ```
- **Parallel fan-out**: Use `sub_query_map` for N independent questions or N chunks that each need LLM analysis.

**Do NOT use `sub_query` for:**
- Simple extraction -- use `search()`, `extract_pattern()`, `find_all()` instead
- Counting, statistics, or aggregation -- use Python directly
- Tasks where you already have enough context in the current REPL to answer with code

**Choosing between `sub_query` and sub-agents (task tool):**
- `sub_query`: Inline recursive decomposition inside a single REPL. Best for programmatic loops over data chunks. Stays within the current analysis context.
- Sub-agents (task tool): Parallel isolated execution with full tool access. Best for independent tasks that need filesystem access, their own context loading, or true parallelism across different data sources.

## Recursive `sub_query` Notes

- `sub_query` is not a flat one-shot completion in this harness; it runs a mini code-execute-observe loop inside a fresh REPL.
- Sub-calls can inspect `ctx`, create variables, run loops, and set `Final` to return results.
- Recursive sub-calls are supported. A configurable max recursion depth limits nesting; once the max depth is reached, sub-calls fall back to a direct model invocation.
- When using reasoning models (e.g. `deepseek-reasoner`) as the sub-query backend, expect longer response times. Configure `sub_query_timeout` accordingly (300s+ recommended for reasoning models).

## Multi-Context Support

Use `context_id` parameter on any tool to work with multiple isolated contexts simultaneously. Default is `"default"`. Example: load a config file into `"config"` and source code into `"source"`, then cross-reference.

## Recipe Pipelines

For repeatable multi-step workflows, use `validate_recipe` + `run_recipe` to execute declarative pipelines with steps like search, chunk, map_sub_query, aggregate, and finalize.

## Binary and PDF Inputs (Important)

When the user asks you to analyze PDFs, images, or other binary files, do **not** treat them as plain text.

- Do not rely on `grep` over raw binary bytes as your primary extraction method.
- If the input is non-text, first convert/extract to text using an appropriate tool path (for example, PDF text extraction or OCR when needed), then load/analyze the extracted text with RLM tools.
- If extraction tooling is unavailable, state the concrete blocker and immediately offer the best executable fallback (install/use an available extractor, or request a text export from the user with exact format).
- Do not ask meta questions like whether to "add this to the system prompt." Focus on completing the user task.

Preferred pattern:
1. Detect file type and extraction viability.
2. Extract to text (or explain exactly why extraction failed).
3. Load extracted content via `load_context` or `load_file_context`.
4. Analyze with `search_context` / `peek_context` / `exec_python`.
