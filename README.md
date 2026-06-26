# seer-annotator

Batch LLM annotation worker for SEER `ExperimentSetup`s.

Takes a pipeline JSON produced by SEER, runs every `(run × paper × question)` cell through an LLM, persists results locally (resumable), and posts `LLMAnswer` records back to SEER.

---

## Install

```bash
pyenv local review   # or: PYENV_VERSION=review
pip install -e ".[dev]"
```

---

## Quick start

```bash
# Run annotation
seer-annotate run pipeline.json

# Filter to specific runs or papers
seer-annotate run pipeline.json --runs 10,11 --papers 42,43

# Dry run — dummy LLM, prints payloads instead of posting
seer-annotate run pipeline.json --dry-run

# Check progress
seer-annotate status pipeline.json
```

Resume after interruption: just re-run the same command. Already-done and posted cells are skipped.

---

## The two-pass pipeline

Every `(paper × question)` cell is processed in two sequential LLM calls with separate models and separate concerns. Knowing which pass to look at cuts debugging time significantly.

```
Paper text + questions
        │
        ▼
┌─────────────────────────────────────────────┐
│  Pass 1 — Reasoning model                   │
│  "What is the answer and why?"              │
│  → free-form reasoning, verbatim quotes,    │
│    confidence judgement                     │
└───────────────────┬─────────────────────────┘
                    │ raw text output
                    ▼
┌─────────────────────────────────────────────┐
│  Pass 2 — Formatting model                  │
│  "Extract that into typed JSON"             │
│  → { value, cited_text, comment,            │
│      confidence } per question              │
└───────────────────┬─────────────────────────┘
                    │ structured JSON
                    ▼
            POST to SEER
```

**If answers are wrong or questions are being misunderstood** → look at Pass 1: the reasoning model, its system prompt, question batching, and the question definitions themselves.

**If the JSON is malformed, values are mis-typed, or `cited_text_verified` is unexpectedly false** → look at Pass 2: the formatting model, its configuration, and the citation verification settings.

---

## Pass 1 — Reasoning model

The reasoning model receives the paper text and a list of questions, then produces free-form output with:

1. A verbatim quote from the paper supporting the answer
2. Step-by-step reasoning
3. A final answer value
4. A confidence score (0–20)

The output is plain text — not JSON. Pass 2 handles the conversion.

### Motivation

Separating reasoning from extraction lets you point a capable (and expensive) model purely at understanding the paper, without forcing it to simultaneously manage JSON syntax. Reasoning errors and extraction errors have different causes, different fixes, and different debugging paths.

### Pass 1 configuration (`runs[*].config`)

| Field | Default | Purpose |
|---|---|---|
| `text_source` | `"full_text"` | `"abstract"` or `"full_text"` (fetches OCR). Papers without OCR are marked `skipped`. |
| `batching` | `"per_question"` | How questions are grouped into LLM calls — see [batching](#batching) below. |
| `temperature` | `0.0` | Passed to the reasoning model. Some models (e.g. gpt-5 family) require `1`. |
| `reasoning_effort` | `null` | Optional — passed to o-series models (e.g. `"high"`). |
| `system_prompt` | `null` | Overrides `runtime.system_prompt` from `settings.toml`. If neither is set, the built-in default is used. |
| `model_params` | `{}` | Extra kwargs forwarded verbatim to the Pass-1 LLM call. Merged last, so they override `temperature` if both are set. |
| `cache` | `false` | Enable prompt caching (Anthropic: adds `cache_control`; OpenAI/Gemini: no-op, handled server-side). |
| `cache_first` | `"text"` | Which block gets the cache marker — `"text"` or `"questions"`. See [caching](#prompt-caching). |
| `cache_ttl` | `"1h"` | Cache TTL for Anthropic: `"5m"` or `"1h"`. |

#### Batching

Controls how questions are grouped into a single LLM call. More questions per call means fewer API calls and better use of shared context (e.g. the paper text); fewer questions per call reduces interference between questions.

| Value | Meaning |
|---|---|
| `"per_question"` | One call per question (default — most isolated) |
| `"all"` | All questions in one call |
| `{"size": N}` | Groups of N questions |
| `[["k1","k2"], ["k3"]]` | Explicit groups by question key |

---

## Pass 2 — Formatting model

The formatting model receives the raw Pass-1 output and restructures it into a typed JSON record per question. Its job is purely mechanical: copy the value, cited text, reasoning comment, and confidence exactly as written — no rephrasing, no summarising, no interpretation.

This produces one entry per question:

```json
{
  "key": "is_paper",
  "value": true,
  "cited_text": "We present...",
  "comment": "The document self-identifies as a paper...",
  "confidence": 18
}
```

### Motivation

A small, cheap model (e.g. `gpt-4o-mini`) is sufficient for this task because it requires no domain knowledge — only reliable JSON syntax and faithful verbatim copying. This makes Pass 2 fast and inexpensive regardless of how capable or expensive the Pass-1 model is.

### Pass 2 configuration (`runs[*].config`)

| Field | Default | Purpose |
|---|---|---|
| `format_model` | `runtime.format_model` | Model for Pass 2. Overrides the global fallback in `settings.toml`. Use the exact deployment name the provider expects. |
| `format_model_provider` | `runtime.format_model_provider` | Provider for the format model. Useful when Pass 1 runs on Azure and Pass 2 on OpenAI. Falls back to the run's own `model_provider` if not set. |
| `format_structured_output` | `true` | When `true`, calls with `response_format: json_object` — enforces valid JSON at the API level and eliminates structural parse errors. Set `false` only if your endpoint does not support `response_format`. |
| `format_model_params` | `{}` | Extra kwargs forwarded to the Pass-2 call. Pass 2 always uses `temperature=0.0`; override here if needed. |

### `cited_text` — single string or array

The Pass-1 model is instructed to quote verbatim spans from the paper. When it cites a single continuous passage (even one with `...` indicating a deliberate skip within it), the formatting model produces a single string:

```json
"cited_text": "We present GIVE, a novel method... Extensive experiments demonstrated..."
```

When the Pass-1 model explicitly cites two or more clearly separate, independent passages from different parts of the paper, the formatting model produces a JSON array — one element per passage:

```json
"cited_text": [
  "The reasoning process involved in GIVE is fully interpretable.",
  "GIVE guides the LLM agent to select the most pertinent expert data (observe)..."
]
```

Do not use an array just because the quote contains `...` — only when the annotation cites genuinely distinct, non-adjacent passages.

### Citation verification

After Pass 2 extracts `cited_text`, it is verified against the source document and stored as `cited_text_verified` (`true` / `false` / `null`). A `false` result means the annotator could not confirm the quote is real text from the paper; it does **not** change the answer value, which is posted to SEER regardless.

Verification uses fuzzy substring matching ([fuzzysearch](https://github.com/taleinat/fuzzysearch)), so minor formatting differences (wrapping quotes, whitespace) do not cause false failures. It handles three citation patterns:

| Pattern | How it is verified |
|---|---|
| Single passage | Fuzzy substring search in the full source |
| `"A ... B"` (ellipsis skip) | A and B are matched independently in order, within a configurable gap |
| Array of passages | Each passage fuzzy-matched independently |

The following verification parameters can be tuned in `settings.toml`:

| Key | Default | Meaning |
|---|---|---|
| `runtime.citation_max_error_rate` | `0.05` | Maximum Levenshtein distance as a fraction of the pattern length. `0.05` = 5% of characters may differ. Increase if the OCR source has many transcription errors. |
| `runtime.citation_max_ellipsis_gap` | `600` | Maximum number of characters allowed between ellipsis parts in the source. Increase for documents where relevant passages are far apart (e.g. full-text papers vs abstracts). |

### What to check when Pass 2 goes wrong

**Parse error / `ExtractionError`**: the formatting model produced malformed JSON. Try switching `format_model` to a more capable or more instruction-following model. If your endpoint does not support `response_format`, set `format_structured_output: false` to fall back to line-by-line parsing with `json-repair` recovery.

**Wrong `value` type** (e.g. string instead of boolean): the format model misread the valid options. Check the question definition — if `options` are missing, the model has no constraint to work against.

**`cited_text_verified: false`**: the quoted span was not found in the source. Common causes:
- Pass-1 model hallucinated a quote that is not in the paper (legitimate failure)
- OCR quality is poor and the source text differs significantly from what the model saw — try increasing `citation_max_error_rate`
- The model cited two separate passages without using array form — check the Pass-1 output in the debug UI and consider adjusting the system prompt

---

## Settings (`settings.toml`)

Settings follow a three-level hierarchy — each level overrides the one above it:

```
code defaults  →  settings.toml [run_defaults]  →  pipeline JSON runs[*].config
```

`[run_defaults]` sets project- or machine-level defaults for any run setting. Values in `runs[*].config` inside the pipeline JSON always win when explicitly present.

### Storage paths (`[runtime]`)

The only settings that cannot be changed per run are storage paths, since they locate the database that ties everything together.

```toml
[runtime]
store_path   = "./.seer_state.db"
p1_dump_dir  = "./tmp"
```

| Key | Default | Meaning |
|---|---|---|
| `store_path` | `"./.seer_state.db"` | Path to the local SQLite state file |
| `p1_dump_dir` | `"./tmp"` | Directory for Pass-1 batch dump files (crash-safe resume) |

### Run defaults (`[run_defaults]`)

These set default values for all per-run settings. Any field present here applies to all runs unless the pipeline JSON's `runs[*].config` explicitly overrides it. All fields are optional — omit them to use the built-in code defaults.

```toml
[run_defaults]
# Execution
concurrency             = 8       # max parallel LLM calls
per_provider_rpm        = null    # omit for unlimited
drop_params             = false

# Citation verification
citation_max_error_rate   = 0.05
citation_max_ellipsis_gap = 600

# Pass-1
text_source             = "full_text"
temperature             = 0.0
reasoning_effort        = null
batching                = "per_question"
model_params            = {}
system_prompt           = null
cache                   = false
cache_first             = "text"
cache_ttl               = "1h"
batch_p1                = false

# Pass-2
format_model            = "gpt-4o-mini"
format_model_provider   = null
format_model_params     = {}
format_structured_output = true
batch_p2                = false
```

| Key | Default | Meaning |
|---|---|---|
| `concurrency` | `8` | Max parallel LLM calls across all papers and groups |
| `per_provider_rpm` | `null` | Max requests-per-minute per provider; `null` = unlimited |
| `drop_params` | `false` | LiteLLM: silently strip unsupported parameters instead of raising |
| `citation_max_error_rate` | `0.05` | Fuzzy match tolerance for citation verification — see [Citation verification](#citation-verification) |
| `citation_max_ellipsis_gap` | `600` | Max source-character gap between ellipsis parts — see [Citation verification](#citation-verification) |
| `text_source` | `"full_text"` | Document source: `"full_text"` (OCR) or `"abstract"` |
| `temperature` | `0.0` | Sampling temperature for Pass-1 |
| `reasoning_effort` | `null` | Reasoning effort for o-series models (e.g. `"high"`) |
| `batching` | `"per_question"` | Question grouping: `"per_question"`, `"all"`, `{"size": N}`, or explicit groups |
| `model_params` | `{}` | Extra kwargs forwarded to the Pass-1 LLM call |
| `system_prompt` | `null` | System prompt for Pass-1 |
| `cache` | `false` | Enable prompt caching — see [Prompt caching](#prompt-caching) |
| `cache_first` | `"text"` | Which block gets the cache marker: `"text"` or `"questions"` |
| `cache_ttl` | `"1h"` | Anthropic cache TTL: `"5m"` or `"1h"` |
| `batch_p1` | `false` | Submit all Pass-1 calls as a single async batch (50% discount on Anthropic/OpenAI/Azure) |
| `format_model` | `"gpt-4o-mini"` | Pass-2 format model |
| `format_model_provider` | `null` | Provider for the format model; falls back to the run's main provider |
| `format_model_params` | `{}` | Extra kwargs forwarded to the Pass-2 LLM call |
| `format_structured_output` | `true` | Use `response_format: json_object` for Pass-2 (disable for endpoints that don't support it) |
| `batch_p2` | `false` | Submit all Pass-2 calls as a single async batch |
| `fail_fast` | `false` | Stop the pipeline on the first LLM or extraction error and raise; when `false` (default), errors are recorded as `extraction_status: "error"` and posted to SEER so the run continues |

### Provider settings

Each `[providers.<name>]` block corresponds to a `model_provider` value in the pipeline JSON.

| Key | Meaning |
|---|---|
| `api_key` | Literal API key (takes precedence over `api_key_env`) |
| `api_key_env` | Environment variable name to read the key from |
| `base_url` | Provider base URL (required for Azure and local/Ollama) |
| `api_version` | API version string (required for Azure) |

```toml
[providers.openai]
api_key_env = "OPENAI_API_KEY"

[providers.anthropic]
api_key_env = "ANTHROPIC_API_KEY"

[providers.azure]
api_key     = "your-azure-key"
base_url    = "https://your-resource.openai.azure.com/"
api_version = "2024-02-01"

[providers.ollama]
api_key  = "ollama"
base_url = "http://localhost:11434/v1"
```

---

## Cost optimisation

### Batch mode

Set `batch_p1 = true` and/or `batch_p2 = true` in `settings.toml` under `[run_defaults]`, or in the pipeline JSON under `runs[*].config`. Anthropic and OpenAI charge 50% less for async batch requests. The CLI submits all requests, polls until done, then processes results.

| Setting | What it batches |
|---|---|
| `batch_p1 = true` | All Pass-1 reasoning calls across all papers |
| `batch_p2 = true` | All Pass-2 format extraction calls |

The most common setup is `batch_p1 = true` alone — the reasoning model is the expensive one; Pass 2 with a small format model is cheap enough to run online.

Batch jobs are resumable: if interrupted while polling, re-running the same command picks up the existing batch from the local store.

**Supported providers:** `anthropic`, `openai`, `azure`.

### Prompt caching

Set `"cache": true` in a run's config block. For Anthropic, this adds `cache_control` markers so a shared prefix (paper text or question list) is reused across calls. OpenAI and local providers handle prefix caching automatically server-side — `cache: true` is a no-op there.

The `cache_first` field controls which block gets the cache marker:

| `cache_first` | Cached block | Best when |
|---|---|---|
| `"text"` (default) | Paper text | Same paper appears in multiple calls (retries, multiple batching groups) |
| `"questions"` | Question block | Same question set runs across many papers — warms on the first paper, hits on every subsequent one |

**With `batch_p1 = true` and `cache = true`:** seer-annotator uses a two-step prewarm strategy (per Anthropic batch caching docs). It submits a single-request prewarm batch first to write the shared prefix into the 1-hour cache, waits for completion, then submits the main batch. Without this, Anthropic's parallel workers each independently write the same cache entry.

`cache_ttl` controls the Anthropic cache duration: `"5m"` (1.25× write cost, for online mode) or `"1h"` (2× write cost, default, for batch mode where requests arrive minutes apart).

### Combined savings (Anthropic, approximate)

| Mode | Input tokens | Output tokens |
|---|---|---|
| Baseline | $5 /M | $15 /M |
| Cache only | ~$0.50 /M (hit) | $15 /M |
| Batch only | $2.50 /M | $7.50 /M |
| Batch + cache | ~$0.25 /M (hit) | $7.50 /M |

---

## Local store

State is kept in a SQLite database (default: `.seer_state.db`).

| Status | Meaning |
|---|---|
| `pending` | Not yet computed |
| `done` | Computed (including error answers), not yet posted to SEER |
| `posted` | Posted to SEER successfully |
| `failed` | Fatal error that prevented saving any answer (e.g. unrecoverable DB or posting failure) |
| `skipped` | Intentionally skipped (e.g. paper filtered out); never posted to SEER |

`posted` and `done` cells are never recomputed on resume.

When the LLM pipeline itself fails for a question (no OCR, API error, parse failure), the answer is still saved as `done` and posted to SEER, but with `extraction_status` set to `"error"` (or `"invalid"` for type-validation failures) and an `extraction_detail` message explaining the cause. The `extraction_status` field has three values:

| `extraction_status` | Meaning |
|---|---|
| `ok` | Value extracted and validated successfully |
| `invalid` | Value failed type validation (e.g. categorical value not in allowed options) |
| `error` | Pipeline failed entirely (no OCR, API error, parse error) |

---

## Debug UI

```bash
seer-annotate ui pipeline.json        # opens at http://127.0.0.1:8765
seer-annotate ui pipeline.json --port 9000
```

| View | What it shows |
|---|---|
| **Overview** | Pipeline config, run list, question list, answer status counts |
| **Answers** | Filterable table of all answers; click any row for full detail |
| **Tokens & Cost** | Per-run token breakdown (input/output/cached) and USD cost totals |

Clicking an answer row opens a side panel with tabs:

| Tab | Content |
|---|---|
| **Structured** | Final typed value, token usage, cost, cited text with verification status, comment |
| **Pass 1 (reason)** | Raw free-form reasoning output from the LLM |
| **Pass 2 (format)** | Formatted extraction output and parsed JSON |
| **Full raw** | Complete `raw_response` JSON blob |
| **OCR text** | Full-text markdown from the OCR cache (if available) |

---

## Authentication

The pipeline JSON includes an `api_token` field containing a DRF token for SEER. This token is static and does not expire. No login step is needed.

---

## Development

```bash
# Run tests (no real LLM calls)
pytest

# Run a specific test file
pytest tests/test_batching.py -v
```

Tests use a mocked SEER server (`respx`) and the built-in dummy LLM provider. Set `SEER_REAL_LLM=1` to make real provider calls.
