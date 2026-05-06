#!/usr/bin/env bash
# Stage 3 smoke: 200-example fine-tune on Set B, 1 epoch, separate run dir.
# Use to verify the loop end-to-end before committing 5+ hours to the real runs.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.train.finetune \
  --config config/config.yaml \
  --train data/processed/set_b_magister.jsonl \
  --run-name student_set_b_smoke \
  --limit 200 \
  --epochs 1 \
  "$@"
