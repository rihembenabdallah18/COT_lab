#!/usr/bin/env bash
# Stage 5a: accuracy + accuracy-with-calculator for all five conditions.
# Reads outputs/generations/<condition>.jsonl produced by Stage 4 and writes
#   outputs/eval_results/accuracy.csv
#   outputs/plots/accuracy_bar.png
#   outputs/runs/05a_accuracy.json
# Pure-CPU; takes seconds. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.eval.accuracy "$@"

echo
echo "=== outputs/eval_results/accuracy.csv ==="
cat outputs/eval_results/accuracy.csv
echo
python -m src.status
