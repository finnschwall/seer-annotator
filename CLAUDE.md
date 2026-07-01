# CLAUDE.md — seer-annotator

Batch LLM annotation worker for SEER (Systematic Evidence Extraction and Research). Reads a pipeline JSON (papers × questions × runs), runs each cell through a two-pass LLM pipeline, stores results in SQLite, posts `LLMAnswer` records back to SEER via REST.

---

## Project layout

```
seer_annotator/
  cli.py            # Click commands: run, pass1, pass2, status, repost, reformat, ui
  config.py         # Pydantic models: PipelineConfig, RunConfig, Settings
  orchestrator.py   # Unified chunked run_pipeline; pass1_pipeline; pass2_pipeline; reformat_pipeline
  store.py          # SQLite: ocr_cache + answers + kv tables
  seer_client.py    # httpx client: fetch OCR markdown, post LLMAnswer
  llm.py            # LiteLLM wrapper → LLMResult (text, usage, cost, latency)
  batching.py       # Question grouping: per_question / all / size-N / explicit
  batch_runner.py   # Shared two-phase engine: _execute_pass1 / _execute_pass2 (online + batch sub-modes)
  caching.py        # Prompt-cache markers per provider (Anthropic/Gemini/OpenAI)
  rate_limiter.py   # Sliding-window RPM throttle, per-provider
  mapping.py        # Build typed LLMAnswer payload by question_type
  annotate/
    engine.py       # Two-pass pipeline: Pass-1 (reason) → Pass-2 (format) → verify
    prompt.py       # Build message lists for Pass-1 and Pass-2
    parse.py        # Parse Pass-2 JSON; json_repair fallback
    verify.py       # Citation fuzzy-match (single / ellipsis / array)
  ui/
    app.py          # FastAPI debug UI (port 8765)
    static/index.html
tests/
  test_integration.py   # Full roundtrip with mocked SEER + dummy LLM
  test_batching.py
  test_mapping.py
  test_store.py
```

---

## Core design decisions

### Two-pass annotation (the central idea)

Every `(run, paper, question_group)` cell goes through **two sequential LLM phases**, implemented as shared engine helpers in `batch_runner.py`:

- **`_execute_pass1`** — reasoning phase (expensive): receives paper text + questions, returns free-form text (evidence quotes, reasoning, answer, confidence 0–20). No JSON required; the model just thinks.
- **`_execute_pass2`** — formatting phase (cheap): receives the Pass-1 text, returns typed JSON with `key`, `value`, `cited_text`, `comment`, `confidence`. Uses `gpt-4o-mini` by default.

Why: decouples domain reasoning errors (blame Pass 1) from JSON extraction errors (blame Pass 2). When debugging, inspect them independently. The two phases are also **separable stages**: the `pass1` command runs only Pass 1, persisting results so a later `pass2` command can run formatting on different infrastructure or at a different time. The `run` command executes both phases in sequence (per chunk) without separating them. Pass-2-only reformat of previously completed answers is a first-class CLI command (`reformat`).

### Configuration hierarchy (three levels, last wins)

```
Pydantic defaults in RunConfig
  → settings.toml [run_defaults]       # machine/project overrides
    → pipeline JSON runs[*].config     # per-run overrides (highest priority)
```

The merge happens in `config.py`. When adding a new tunable, add it to `RunConfig` with a sensible default and it propagates automatically.

### SQLite state and resumability

`store.py` tracks every `(run_id, paper_id, version_id)` cell:
- Status flow via `run`: `pending` → `done` → `posted` (or `skipped` / `error`)
- Status flow via `pass1` / `pass2`: `pending` → `pass1_done` → `done` → `posted`
  - `pass1_done` is written only by the `pass1` command (not by `run`); it indicates Pass-1 text is stored and ready for Pass-2 formatting
- On restart: `done`/`posted` cells are skipped — no recomputation; `pass1_done` cells are skipped by `pass1` but picked up by `pass2`
- OCR markdown is fetched once and cached in `ocr_cache`; `batch_runner` stores batch job IDs in the `kv` table

Do not assume work is idempotent above the store layer; the store is the idempotency boundary.

### Batching modes

Controlled by `batching` in `RunConfig`:

| Value | Behavior |
|-------|----------|
| `"per_question"` | One LLM call per question (default) |
| `"all"` | All questions in one call |
| `{"size": N}` | Groups of N questions |
| `[["q1","q2"], ["q3"]]` | Explicit groups |

Batching affects prompt construction (`prompt.py`) and how `mapping.py` splits the Pass-2 response back into per-question answers.

`chunk_papers` (default `10`) controls how many papers `run` processes in each chunk: for each chunk it completes all of Pass 1, then all of Pass 2, then posts. Smaller chunks mean earlier posting and finer crash recovery; larger chunks increase infra-separation between phases. Override per-run in the pipeline JSON or at invocation with `--chunk-papers`. The `pass1` / `pass2` commands always process all papers at once (they are already single-phase by design).

### Provider abstraction

All LLM calls go through `llm.py` which wraps LiteLLM. Model strings are `provider/model` (e.g. `openai/gpt-4o`, `anthropic/claude-3-5-sonnet-20241022`). Provider credentials live in `settings.toml [providers.<name>]` — `api_key_env` or literal `api_key`, plus optional `base_url` and `api_version` for Azure/local.

### Async batch mode (`batch_p1` / `batch_p2`)

`batch_p1` and `batch_p2` drive the two shared engine helpers `_execute_pass1` and `_execute_pass2` in `batch_runner.py`. Each helper independently selects online or async-batch sub-mode based on its flag. This means batch mode is honored in all three commands (`run`, `pass1`, `pass2`): `batch_p1` affects the Pass-1 phase wherever it runs, and `batch_p2` affects the Pass-2 phase. Batch IDs are stored in `kv` so polling survives restarts.

### Citation verification

After Pass 2, `verify.py` fuzzy-matches `cited_text` against the source paper. Supports plain string, `"A ... B"` ellipsis pattern (A and B matched independently in order), and arrays of passages. Result is `cited_text_verified: true/false/null` — verification failure does **not** invalidate the answer.

---

## Key data models (`config.py`)

- **`PipelineConfig`** — the input JSON from SEER: `papers`, `questions`, `runs`
- **`RunConfig`** — all per-run tunables (concurrency, models, batching, caching, citation params, fail_fast)
- **`Settings`** — loaded from `settings.toml`: `runtime`, `run_defaults`, `providers`
- **`LLMResult`** (in `llm.py`) — normalized LLM response: `text`, `reasoning_content`, `usage`, `cost`, `latency_ms`

The `LLMAnswer` payload sent to SEER is built in `mapping.py` and uses typed value fields (`value_boolean`, `value_text`, `value_categorical`, `value_categorical_multi`) determined by `question_type`.

---

## CLI commands

```
seer-annotate run pipeline.json        # main annotation loop (Pass-1 then Pass-2 per chunk)
seer-annotate pass1 pipeline.json      # run only Pass-1; persist results as pass1_done
seer-annotate pass2 pipeline.json      # run only Pass-2 on pass1_done cells; auto-post unless --no-post
seer-annotate status pipeline.json    # token/cost summary from store
seer-annotate repost pipeline.json    # re-post stored answers to SEER
seer-annotate reformat pipeline.json  # re-run Pass-2 only on done/posted answers (switch format model)
seer-annotate ui pipeline.json        # debug UI at http://127.0.0.1:8765
```

---

## Testing

Tests use **respx** to mock SEER HTTP endpoints and a built-in dummy LLM (no real API calls unless `SEER_REAL_LLM=1`). Run with `pytest`. The integration test (`test_integration.py`) covers the full roundtrip including OCR fetch, annotation, and posting.

---

## Things to watch out for

- `mapping.py` must handle every `question_type` defined in SEER — check here first when adding new question types.
- `parse.py` uses `json_repair` as a fallback; if answers look wrong, check whether Pass-2 is producing malformed JSON and whether the repair is silently corrupting values.
- The `version_id` (not `question_id`) is the dedup key for answers in the store — don't confuse them.
- `fail_fast = false` (default) stores errors and continues; set `true` only for debugging single cells.
- Prompt-cache markers in `caching.py` are provider-specific and fragile — test against real APIs when changing message structure.
