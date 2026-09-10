#!/usr/bin/env bash
# Every arm of PREREGISTRATION.md, in the order the pre-registration declares.
# Arm A must finish before the class exists; the class must exist before arm C
# or arm F can run.
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/brandonin/Documents/typed-crystallization-networks/.venv/bin/python

echo "=== arm A: certify at the training resolutions ==="
$PY -u run.py enumerate --resolution 24 --screens 6
$PY -u run.py enumerate --resolution 32 --screens 12    # the section 33 replication

echo "=== construct the class ==="
$PY -u run.py construct

echo "=== arm B: enumerate at the held-out resolutions ==="
$PY -u run.py enumerate --resolution 16 --screens 6
$PY -u run.py enumerate --resolution 40 --screens 6
$PY -u run.py enumerate --resolution 48 --screens 6

echo "=== arm C: instantiate at the held-out resolutions ==="
$PY -u run.py instantiate --resolution 16 --screens 6
$PY -u run.py instantiate --resolution 40 --screens 6
$PY -u run.py instantiate --resolution 48 --screens 6

echo "=== arm F, F-a, null control ==="
$PY -u run.py armf
$PY -u run.py refusal
$PY -u run.py null

echo "=== criteria ==="
$PY -u check.py
