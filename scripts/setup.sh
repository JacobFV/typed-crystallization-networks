#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --locked --extra test
npm --prefix generators/computer/engine ci --ignore-scripts
