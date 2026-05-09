#!/usr/bin/env bash
# Stage 3: fine-tune FLAN-T5-base on Set C (calculator-corrected filter).
# Set C is process-aware - rejects right-answer-wrong-arithmetic chains.
# Pass --resume to pick up after a session timeout.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.train.finetune \
  --config config/config.yaml \
  --train data/processed/set_c_calculator.jsonl \
  --run-name student_set_c \
  "$@"
