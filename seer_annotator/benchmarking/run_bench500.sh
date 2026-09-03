#!/usr/bin/env bash
set -Eeuo pipefail

# One provider lane of the 500-abstract Phase-2 formatting benchmark.
#
# Both lanes write to the same SQLite file. That is safe: execution rows are
# claimed atomically (store.claim_execution) and the store runs in WAL mode, so
# two processes never contend for the same cell. Run one lane per provider so
# their rate limits stay independent:
#
#   LANE=ollama bash seer_annotator/benchmarking/run_bench500.sh
#   LANE=azure  bash seer_annotator/benchmarking/run_bench500.sh
#
# Both are resumable: completed executions are skipped, so re-running the same
# command picks up missing work. Stored errors are retried only with
# RETRY_ERRORS=1.

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"

SEER_ANNOTATE="${SEER_ANNOTATE:-seer-annotate}"
SEER_ROOT="${SEER_ROOT:-/home/ies/schwall/sysreview/SEER-prod}"
SETTINGS="${SETTINGS:-${REPO_ROOT}/working_dir/settings.toml}"
MODELS="${MODELS:-${SCRIPT_DIR}/bench500_models.toml}"

SOURCE_RUN="${SOURCE_RUN:-36}"
PAPER_COUNT="${PAPER_COUNT:-500}"
SEED="${SEED:-20260825}"
DATASET="${DATASET:-abstract-ic-run${SOURCE_RUN}-${PAPER_COUNT}-seed${SEED}}"

RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/benchmarking/bench500}"
BENCHMARK_DB="${BENCHMARK_DB:-${RESULT_DIR}/phase2-500.sqlite}"

LANE="${LANE:-ollama}"
case "${LANE}" in
  ollama)
    LANE_MODELS=(deepseek-v4-flash-nothink gemma4-31b "qwen3.5-397b" deepseek-v4-pro)
    CONCURRENCY="${CONCURRENCY:-4}"
    RPM="${RPM:-60}"
    ;;
  azure)
    LANE_MODELS=(gpt-5-mini gpt-5-nano)
    CONCURRENCY="${CONCURRENCY:-8}"
    RPM="${RPM:-120}"
    ;;
  *)
    printf 'unknown LANE %q (expected ollama or azure)\n' "${LANE}" >&2
    exit 2
    ;;
esac

LOG_FILE="${LOG_FILE:-${RESULT_DIR}/${DATASET}-${LANE}.log}"
RETRY_ERRORS="${RETRY_ERRORS:-0}"
IMPORT="${IMPORT:-1}"

mkdir -p "${RESULT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

printf 'Lane %s started: %s\n' "${LANE}" "$(date --iso-8601=seconds)"
printf 'Dataset: %s\nSQLite: %s\nModels: %s\n' \
  "${DATASET}" "${BENCHMARK_DB}" "${LANE_MODELS[*]}"

# Idempotent; only the lane launched first needs it. IMPORT=0 skips it so the
# second lane never opens the source database at all.
if [[ "${IMPORT}" == "1" ]]; then
  "${SEER_ANNOTATE}" benchmark import-db \
    --seer-root "${SEER_ROOT}" \
    --benchmark-db "${BENCHMARK_DB}" \
    --dataset "${DATASET}" \
    --run "${SOURCE_RUN}:${PAPER_COUNT}" \
    --seed "${SEED}"
fi

retry_args=()
if [[ "${RETRY_ERRORS}" == "1" ]]; then
  retry_args+=(--retry-errors)
fi

for model_name in "${LANE_MODELS[@]}"; do
  printf '\nRunning model: %s (%s)\n' "${model_name}" "$(date --iso-8601=seconds)"
  "${SEER_ANNOTATE}" benchmark run \
    --benchmark-db "${BENCHMARK_DB}" \
    --dataset "${DATASET}" \
    --settings "${SETTINGS}" \
    --models "${MODELS}" \
    --model "${model_name}" \
    --concurrency "${CONCURRENCY}" \
    --rpm "${RPM}" \
    "${retry_args[@]}"
done

printf '\nLane %s finished: %s\n' "${LANE}" "$(date --iso-8601=seconds)"
"${SEER_ANNOTATE}" benchmark status --benchmark-db "${BENCHMARK_DB}" --dataset "${DATASET}"
