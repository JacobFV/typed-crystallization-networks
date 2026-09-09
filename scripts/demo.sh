#!/usr/bin/env bash
# Run every capability this repository can actually demonstrate, on the current
# tree, and print each number next to the reference it has to beat.
#
#   scripts/demo.sh                    the quick pass, roughly four minutes
#   scripts/demo.sh --full             larger seeds and wider inputs
#   scripts/demo.sh --only depth       one demonstration
#   scripts/demo.sh --list             the names
#
# Artifacts land under artifacts/demo/<name>/, with artifacts/demo/summary.json
# and artifacts/demo/summary.md holding the table. Exit status is non-zero if
# any demonstration fails to reproduce.
#
# `.venv/bin/tcn` is a console script whose shebang points at the interpreter
# the wheel was first built against, which does not exist on this host; every
# entry point is reached through `python -m tcn` instead. See STATUS.md.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m tcn demo --out "${TCN_DEMO_OUT:-artifacts/demo}" "$@"
