#!/usr/bin/env bash
# Stage 4: greedy inference on GSM8K test set for all three conditions.
# Run on the same T4 session where training completed (or after restoring
# checkpoints). Each condition is resumable -- safe to re-run after timeout.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== baseline ==="
python -m src.inference.generate --condition baseline

echo "=== student_set_b ==="
python -m src.inference.generate --condition student_set_b

echo "=== student_set_a ==="
python -m src.inference.generate --condition student_set_a

echo "=== all done ==="
ls -lh outputs/generations/
