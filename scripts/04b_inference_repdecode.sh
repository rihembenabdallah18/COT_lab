#!/usr/bin/env bash
# Stage 4 diagnostic: re-run inference on the SAME checkpoints with anti-loop
# decoding tweaks (repetition_penalty + no_repeat_ngram_size). Output is written
# to outputs/generations/{condition}_repdecoded.jsonl so the original greedy
# generations are preserved for comparison.
#
# Goal: determine whether the low post-distillation accuracy is a decoding
# pathology (greedy loops) or a genuine training failure.
set -euo pipefail
cd "$(dirname "$0")/.."

REP_PENALTY=1.3
NO_REPEAT_NGRAM=4
SUFFIX="_repdecoded"

for COND in baseline student_set_b student_set_a; do
  echo "=== ${COND} (rep_penalty=${REP_PENALTY}, no_repeat_ngram=${NO_REPEAT_NGRAM}) ==="
  python -m src.inference.generate \
    --condition "${COND}" \
    --repetition-penalty "${REP_PENALTY}" \
    --no-repeat-ngram-size "${NO_REPEAT_NGRAM}" \
    --out-suffix "${SUFFIX}"
done

echo "=== all done ==="
ls -lh outputs/generations/
