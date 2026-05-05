#!/usr/bin/env bash
# Stage 1: download GSM8K + Ho et al. teacher CoTs.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.data.download "$@"
