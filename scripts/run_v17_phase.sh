#!/usr/bin/env bash
set -euo pipefail

V="$1"
SHARD="$2"
TOKENIZER="$3"
VAL="$4"
MILESTONES_CSV="$5"
DATA_SEED_OFFSET="$6"
BOUNDARY_M="$7"
RESUME_PATH="${8:-}"\nPREFIX="${9:-v17}"

mkdir -p "work/${V}" "metrics/${V}"
CURRENT="work/${V}/current.pt"

if [[ -n "${RESUME_PATH}" ]]; then
  cp "${RESUME_PATH}" "${CURRENT}"
fi

IFS=',' read -ra MILESTONES <<< "${MILESTONES_CSV}"
FIRST=1

for M in "${MILESTONES[@]}"; do
  TARGET=$((M * 1000000))
  NEXT="work/${V}/next.pt"
  METRIC_DIR="metrics/${V}/${M}m"
  mkdir -p "${METRIC_DIR}"

  EXTRA_ARGS=()
  if [[ -f "${CURRENT}" ]]; then
    EXTRA_ARGS+=(--resume "${CURRENT}")
  fi
  if [[ "${FIRST}" == "1" && -n "${RESUME_PATH}" ]]; then
    EXTRA_ARGS+=(--reset-data-cursor)
  fi

  echo "=== variant=${V} milestone=${M}M target=${TARGET} ==="
  set -o pipefail
  python -u train_v17_chunk.py \
    --tokens "${SHARD}" \
    --tokenizer "${TOKENIZER}" \
    --variant "${V}" \
    --target-tokens "${TARGET}" \
    --data-seed-offset "${DATA_SEED_OFFSET}" \
    --seq-len 128 --batch-size 12 --lr 0.00028 --seed 71 \
    "${EXTRA_ARGS[@]}" \
    --save "${NEXT}" \
    --report "${METRIC_DIR}/train.json" \
    | tee "${METRIC_DIR}/train.log"

  python -u eval_v17.py "${NEXT}" \
    --tokenizer "${TOKENIZER}" \
    --val "${VAL}" \
    --chars 100000 \
    --out "${METRIC_DIR}/eval.json"

  rm -f "${CURRENT}"
  mv "${NEXT}" "${CURRENT}"
  FIRST=0
done

FINAL="work/${V}/${PREFIX}-${V}-${BOUNDARY_M}m.pt"
mv "${CURRENT}" "${FINAL}"
echo "phase_complete variant=${V} boundary=${BOUNDARY_M}M checkpoint=${FINAL}"
