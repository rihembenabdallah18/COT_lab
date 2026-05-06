#!/usr/bin/env bash
# Stage 3: fine-tune FLAN-T5-base on Set A (no filter).
# Run on a T4 (Kaggle/Colab). Pass --resume to pick up after a session timeout.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.train.finetune \
  --config config/config.yaml \
  --train data/processed/set_a_nofilter.jsonl \
  --run-name student_set_a \
  "$@"
