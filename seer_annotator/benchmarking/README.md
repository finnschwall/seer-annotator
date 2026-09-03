# Phase-2 formatting benchmark

This package benchmarks the formatting stage independently from SEER. It has
three separate stages:

1. `import-db` reads selected experiment runs from a configured SEER database
   and freezes their Pass-1 text, question groups and source text in local
   SQLite.
2. `run` reads that SQLite file only and stores each model's raw response,
   parsed answers, diagnostics and usage. Completed cells are resumable;
   `--retry-errors` is required to retry failures.
3. `evaluate` reads stored outputs only. It never calls a model or opens the
   source checkout. Any stored model can be selected as the reference.

The source database is opened in an asserted PostgreSQL read-only transaction;
archived SQLite state files use `mode=ro`. Source credentials are redacted
before provenance is stored. The benchmark database is the only writable
database involved. Model configuration fingerprints make changed model,
temperature or LiteLLM parameters distinct experiments.

## Commands

```bash
seer-annotate benchmark list-db-runs \
  --seer-root /home/ies/schwall/sysreview/SEER-prod

seer-annotate benchmark import-db \
  --seer-root /home/ies/schwall/sysreview/SEER-prod \
  --benchmark-db benchmarking/phase2.sqlite \
  --dataset abstract-ic-v1 --run 36:200 --run 37:200 --seed 20260825

seer-annotate benchmark run \
  --benchmark-db benchmarking/phase2.sqlite --dataset abstract-ic-v1 \
  --settings working_dir/settings.toml \
  --models seer_annotator/benchmarking/models.toml.example \
  --model reference --concurrency 8 --rpm 120
seer-annotate benchmark run \
  --benchmark-db benchmarking/phase2.sqlite --dataset abstract-ic-v1 \
  --settings working_dir/settings.toml \
  --models seer_annotator/benchmarking/models.toml.example \
  --model candidate --concurrency 32

seer-annotate benchmark status --benchmark-db benchmarking/phase2.sqlite --dataset abstract-ic-v1
seer-annotate benchmark evaluate --benchmark-db benchmarking/phase2.sqlite \
  --dataset abstract-ic-v1 --reference reference --candidates all \
  --quote-threshold .90 --json-out report.json --csv-out report.csv
```

Use `--case-kind full_text` to freeze original OCR text (and
`--allow-current-source` only when explicitly accepting weaker current-source
provenance). The case schema stores `text_kind`, ordered question objects and
the Pass-1 group, so the same database format supports the current four-question
abstract/IC task and future 20-question/full-text questionnaires.

## Metrics

The primary score is strict group accuracy: emitted keys must occur exactly
once and in the requested order, and every answer must match the reference's
mechanical status, type-sensitive value, confidence, exact comment and fuzzy
citations. Categorical multi-values are compared unordered. Quotes are
NFKC/casefold/whitespace normalized and optimally paired one-to-one; counts must
match and each similarity must meet the threshold (default `.90`). Empty
quotes are compared explicitly. Reports include answer accuracy, native
JSON/schema/fallback/repair/missing/duplicate/unexpected diagnostics, coverage,
errors, usage, cost and latency, with breakdowns by source run, question key/type
and group size. Component rates for status, value, confidence, comment and
citation are retained in both aggregate reports and answer rows. The existing
deterministic citation verifier is reported separately (answers with citations,
verified, failed) and does not alter model-vs-model accuracy.

Separately from reference comparison, every model is scored on **fabrication**,
which needs no reference — it compares the model's own answer against the frozen
Pass-1 text. `pass1_blocks_absent` counts the answers whose question Pass-1 never
wrote an `--- ANSWER: <key> ---` block for; that is a property of the cases, so
it is identical for every model and doubles as a sanity check.

- `ghost_answers` — the model reported `ok`/`unmappable` for one of those. It
  invented an answer. `ghost_answers_with_value` is the subset carrying a
  non-null value, i.e. a fabrication that would reach the database.
- `ghost_rate` — `ghost_answers / pass1_blocks_absent`: of the answers this
  model could have fabricated, the fraction it did.
- `false_absent` / `false_absent_rate` — the opposite error: the block IS present
  and the model said `absent`. Data loss, not fabrication, so it is never folded
  into the ghost numbers.
- `parse_recovered` — answers stored as `absent` that the current parser reads
  as a real status. An older parser wrote `absent` whenever it failed to extract
  a key, which makes a model look as though it claimed `absent` when it did not.
  The ghost/false-absent metrics therefore read the model's claim from a
  re-parse of `raw_response`, the same source the accuracy metrics use; this
  counter is the gap. Charging a model for a parser limitation ranks it wrongly.

The annotate pipeline enforces the same rule deterministically — see
`annotate/scope.py` — so a ghost answer can never reach SEER. The benchmark
deliberately records the RAW model-reported status so this metric keeps
measuring the model rather than the guard.

Reference-invalid groups are excluded with a count. Candidate missing/error
groups count as failures.

## Smoke test

For a quick supervised check, create a fresh database and import five papers
from run 36:

```bash
seer-annotate benchmark import-db \
  --seer-root /home/ies/schwall/sysreview/SEER-prod \
  --benchmark-db /tmp/phase2-smoke.sqlite --dataset smoke-5 \
  --run 36:5 --seed 20260825
seer-annotate benchmark run --benchmark-db /tmp/phase2-smoke.sqlite \
  --dataset smoke-5 --settings working_dir/settings.toml \
  --models seer_annotator/benchmarking/models.toml.example --model reference
seer-annotate benchmark run --benchmark-db /tmp/phase2-smoke.sqlite \
  --dataset smoke-5 --settings working_dir/settings.toml \
  --models seer_annotator/benchmarking/models.toml.example --model candidate
seer-annotate benchmark evaluate --benchmark-db /tmp/phase2-smoke.sqlite \
  --dataset smoke-5 --reference reference --candidates candidate \
  --json-out /tmp/phase2-smoke.json --csv-out /tmp/phase2-smoke.csv
```

The example reference is `ollama/deepseek-v4-flash:cloud` through Ollama's
direct cloud endpoint; the candidate is `openai/kit.gemma4-31b-it`. Both send
the frozen abstract and Pass-1 text to their configured provider. Re-run each
`benchmark run` command to confirm all five completed executions are skipped,
then re-run `evaluate` with another `--quote-threshold` to confirm evaluation
is fully offline.

Confirm five terminal executions per model, raw/parsed/usage rows and a
JSON/CSV report. Re-running both model commands should show all five cells
skipped; changing only the evaluation quote threshold should produce a new
report with no model calls.

## 500-abstract benchmark

`bench500_models.toml` defines six models across two provider lanes with
independent rate limits. Both lanes write to the same SQLite file, which is safe:
execution rows are claimed atomically and the store runs in WAL mode, so two
processes never contend for the same cell.

```bash
seer-annotate benchmark import-db \
  --seer-root /home/ies/schwall/sysreview/SEER-prod \
  --benchmark-db benchmarking/bench500/phase2-500.sqlite \
  --dataset abstract-ic-run36-500-seed20260825 --run 36:500 --seed 20260825

IMPORT=0 LANE=ollama nohup bash seer_annotator/benchmarking/run_bench500.sh &
IMPORT=0 LANE=azure  nohup bash seer_annotator/benchmarking/run_bench500.sh &
```

`LANE=ollama` runs `deepseek-v4-flash-nothink`, `gemma4-31b`, `qwen3.5-397b` and
`deepseek-v4-pro` at concurrency 4 / 60 RPM. `LANE=azure` runs `gpt-5-mini` and
`gpt-5-nano` at concurrency 8 / 120 RPM. `IMPORT=1` (the default) snapshots the
dataset first; pass `IMPORT=0` on the second lane so it never opens the source
database. Both lanes are resumable — completed executions are skipped — and
stored errors are retried only with `RETRY_ERRORS=1`. Override `SOURCE_RUN`,
`PAPER_COUNT`, `SEED`, `CONCURRENCY`, `RPM`, `BENCHMARK_DB` or `RESULT_DIR` from
the environment.

Two model-config details are not obvious and are load-bearing:

- **Reasoning off for `deepseek-v4-flash-nothink`.** LiteLLM maps any
  `reasoning_effort` outside `{low, medium, high}` to Ollama's `"think": false`.
  With reasoning on, this model emitted valid JSON only 80% of the time at a p95
  latency of 326s; off, it is 98.8% and 2.0s.
- **`temperature = 1.0` for the Azure gpt-5 deployments.** They reject 0.0
  outright — 1 is the only accepted value — so they cannot share the 0.0 the
  Ollama configs use.

`analyze.py` in the result directory is a reference-free companion report:
fabrication and data-loss counts, wire reliability recomputed from stored raw
responses with the current parser, cost/latency, and whether enforcing the
Pass-1 rule would move a paper's IC gate. `benchmark evaluate` cannot score its
own reference, so use `analyze.py` when you need every model on one page.
