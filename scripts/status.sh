#!/usr/bin/env bash
# Convenience wrapper. Prints a one-screen view of every stage's status,
# reading run-cards under outputs/runs/.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.status "$@"
