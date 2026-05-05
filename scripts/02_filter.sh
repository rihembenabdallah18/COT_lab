#!/usr/bin/env bash
# Stage 2: build Set A (no filter) and Set B (Magister filter).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.data.filter "$@"
