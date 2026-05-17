#!/usr/bin/env bash
# Stage 5b: ReCEval scoring for all conditions (including online-calculator variants).
# Prerequisites: Stage 4 inference complete.
#
# NOTE: ReCEval can be run even when distilled students score below baseline on
# accuracy — it is the primary diagnostic for understanding *why* accuracy is
# low (reasoning quality vs answer extraction vs recipe issues). The STOP in
# Stage 5a gates final paper conclusions, not diagnostic runs.
#
# Smoke test first (20 examples, prints step details):
#   bash scripts/05b_receval.sh --smoke 20
#
# Full run:
#   bash scripts/05b_receval.sh
#
# If a condition exceeds ~3 hours, cap at 500 examples:
#   bash scripts/05b_receval.sh --max-examples 500
#
# Outputs written incrementally — safe to re-run after a session timeout.
# Each condition appends to outputs/eval_results/ and regenerates the
# summary CSV and violin plot.
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA_ARGS=("$@")

CONDITIONS=(
  baseline
  student_direct_ft
  student_set_a
  student_set_b
  student_set_c
  student_set_a_oc
  student_set_b_oc
  student_set_c_oc
)

# If smoke mode, run only student_set_b (cheapest meaningful check)
if [[ "${1:-}" == "--smoke" ]]; then
  echo "=== SMOKE TEST: student_set_b, ${2:-20} examples ==="
  python -m src.eval.receval.score_chain \
    --condition student_set_b \
    --smoke "${2:-20}"
  exit 0
fi

for COND in "${CONDITIONS[@]}"; do
  echo "=== ${COND} ==="
  python -m src.eval.receval.score_chain \
    --condition "${COND}" \
    "${EXTRA_ARGS[@]}"
done

echo
echo "=== outputs/eval_results/receval_summary.csv ==="
cat outputs/eval_results/receval_summary.csv
echo
python -m src.status
