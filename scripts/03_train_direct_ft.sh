#!/usr/bin/env bash
# Stage 3: fine-tune FLAN-T5-base on Direct FT (Q -> A only, no CoT).
# Run first as the cheapest sanity check on the v2 recipe.
# Pass --resume to pick up after a session timeout.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.train.finetune \
  --config config/config.yaml \
  --train data/processed/direct_ft.jsonl \
  --run-name student_direct_ft \
  "$@"
