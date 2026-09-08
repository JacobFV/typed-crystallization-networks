#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m pytest -q
npm --prefix generators/computer/engine test
npm --prefix generators/computer/engine run typecheck
