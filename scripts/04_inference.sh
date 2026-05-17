#!/usr/bin/env bash
# Stage 4: inference on the GSM8K test set for all five conditions.
# Each condition is resumable - safe to re-run after a session timeout.
set -euo pipefail
cd "$(dirname "$0")/.."

CONDITIONS=(
  baseline
  student_direct_ft
  student_set_a
  student_set_b
  student_set_c
)

for COND in "${CONDITIONS[@]}"; do
  echo "=== ${COND} ==="
  python -m src.inference.generate --condition "${COND}"
done

echo "=== all done ==="
ls -lh outputs/generations/
python -m src.status
