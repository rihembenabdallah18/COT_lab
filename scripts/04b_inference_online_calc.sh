#!/usr/bin/env bash
# Stage 4b: online calculator decoding for CoT student conditions.
#
# Runs the same checkpoints as Stage 4 but with token-by-token greedy
# generation that intercepts each completed equation and injects the correct
# result before the next token is sampled.
#
# Output: outputs/generations/student_set_{a,b,c}_oc.jsonl
# Runtime: ~40 min per condition on T4 (no batching). Run after Stage 4.
set -euo pipefail
cd "$(dirname "$0")/.."

for COND in student_set_a_oc student_set_b_oc student_set_c_oc; do
  echo "=== ${COND} ==="
  python -m src.inference.generate --condition "${COND}"
done

echo "=== all done ==="
ls -lh outputs/generations/*_oc.jsonl
python -m src.status
