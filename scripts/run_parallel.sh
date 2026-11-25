#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: run_parallel.sh CONFIG [NUM_SHARDS] [RUN_DIR]}"
NUM_SHARDS="${2:-}"
RUN_DIR="${3:-}"

if [[ -z "${NUM_SHARDS}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_SHARDS="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    NUM_SHARDS=1
  fi
fi
if [[ -z "${NUM_SHARDS}" || "${NUM_SHARDS}" -lt 1 ]]; then
  NUM_SHARDS=1
fi
if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="outputs/$(basename "${CONFIG}" .yaml)"
fi

PYTHON="${PYTHON:-python}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
mkdir -p "${RUN_DIR}"
echo "Launching ${NUM_SHARDS} shard(s) for ${CONFIG} -> ${RUN_DIR}"

pids=()
for ((i = 0; i < NUM_SHARDS; i++)); do
  CUDA_VISIBLE_DEVICES="${i}" "${PYTHON}" scripts/run_eval.py \
    --config "${CONFIG}" \
    --shard-index "${i}" \
    --num-shards "${NUM_SHARDS}" \
    --run-dir "${RUN_DIR}" \
    ${EXTRA_ARGS} \
    > "${RUN_DIR}/shard-${i}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "At least one shard failed. Check ${RUN_DIR}/shard-*.log" >&2
  exit "${status}"
fi

"${PYTHON}" scripts/merge_shards.py --run-dir "${RUN_DIR}"
"${PYTHON}" scripts/collect_results.py
