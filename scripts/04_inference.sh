#!/usr/bin/env bash
# Stage 4: inference on the GSM8K test set for all six conditions.
# `baseline` uses beam=4 (v2 recipe); `baseline_greedy` uses greedy decoding
# (beam=1, no penalties) to match Ho et al.'s zero-shot evaluation protocol.
# All other decoding flags come from config/config.yaml. Each condition is
# resumable - safe to re-run after a Kaggle session timeout. Each writes a
# Stage-4 run-card to outputs/runs/04_inference_<condition>.json.
set -euo pipefail
cd "$(dirname "$0")/.."

CONDITIONS=(
  baseline
  baseline_greedy
  student_direct_ft
  student_set_b
  student_set_c
  student_set_a
)

for COND in "${CONDITIONS[@]}"; do
  echo "=== ${COND} ==="
  python -m src.inference.generate --condition "${COND}"
done

echo "=== all done ==="
ls -lh outputs/generations/
python -m src.status
