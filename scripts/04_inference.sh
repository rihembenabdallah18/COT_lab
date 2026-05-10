#!/usr/bin/env bash
# Stage 4: inference on the GSM8K test set for all five conditions.
# Decoding flags (beam=4, no_repeat_ngram_size=4, repetition_penalty=1.15,
# max_new_tokens=512) come from config/config.yaml and are baked into
# src/inference/generate.py defaults. Each condition is resumable - safe to
# re-run after a Kaggle session timeout. Each writes a Stage-4 run-card to
# outputs/runs/04_inference_<condition>.json.
set -euo pipefail
cd "$(dirname "$0")/.."

CONDITIONS=(
  baseline
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
