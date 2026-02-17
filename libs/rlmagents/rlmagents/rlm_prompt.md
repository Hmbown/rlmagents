# RLM (Recursive Language Model) Tools

You have access to RLM context isolation tools for structured analysis and evidence-backed reasoning. Use these when working with large data, complex analysis, or when you need provenance tracking.

## When to Use RLM Tools

- **Large files**: When `read_file` returns very large content, it may be auto-loaded into an RLM context. Use `search_context` / `peek_context` to explore it efficiently.
- **Structured analysis**: When you need to search, extract patterns, run computations, or build evidence for conclusions.
- **Multi-source analysis**: Use multiple `context_id` values to analyze different data in parallel.
- **Evidence-backed reasoning**: When answers need citations, provenance, and confidence tracking.

## Core Workflow

1. **Load**: Use `load_context` to load text/data into an isolated context.
2. **Explore**: Use `peek_context`, `search_context`, or `semantic_search` to understand the data.
3. **Analyze**: Use `exec_python` to run Python code with 100+ built-in helpers (regex, stats, extraction, etc.).
4. **Track**: Evidence is recorded automatically. Use `get_evidence` to review what you've found.
5. **Reason**: Use `think` to structure sub-steps. Use `evaluate_progress` to assess confidence.
6. **Conclude**: Use `finalize` to produce a cited answer.

## Available REPL Helpers (inside `exec_python`)

When executing Python in a context, these helpers are pre-loaded:

**Navigation**: `peek(start, end)`, `lines(start, end)`, `head(n)`, `tail(n)`
**Search**: `search(pattern)`, `grep(pattern)`, `semantic_search(query)`, `find_all(pattern)`
**Extraction**: `extract_numbers()`, `extract_emails()`, `extract_urls()`, `extract_dates()`, `extract_functions(lang)`, `extract_classes(lang)`, `extract_imports(lang)`, `extract_json_objects()`
**Statistics**: `word_count()`, `line_count()`, `char_count()`, `word_frequency()`, `ngrams(n)`
**Text ops**: `replace_all(old, new)`, `between(start, end)`, `split_by(delim)`, `chunk(size, overlap)`
**Citations**: `cite(snippet, line_range, note)` -- records evidence with provenance
**Sub-queries**: `sub_query(prompt, context_slice)` -- delegate to a sub-LLM
**Context mutation**: `ctx_append(text)`, `ctx_set(text)` -- modify context in place

The full context is available as `ctx` variable.

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
3. Load extracted content via `load_context`.
4. Analyze with `search_context` / `peek_context` / `exec_python`.
