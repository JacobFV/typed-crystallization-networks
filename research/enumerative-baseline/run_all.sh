#!/usr/bin/env bash
# Reproduce every arm in RESULTS.md, one at a time so wall clock stays comparable.
set -u
cd "$(dirname "$0")"
PY=../../.venv/bin/python
$PY mixed_task.py                    > out/mixed.log 2>&1
$PY joint_task.py                    > out/joint.log 2>&1
$PY mixed_constants.py               > out/mixed_constants.log 2>&1
$PY scaling.py generator 1 2 3 4 5 6 > out/scaling_generator.log 2>&1
$PY scaling.py hard      3 4 5 6     > out/scaling_hard.log 2>&1
$PY noise.py 2                       > out/noise.log 2>&1
$PY headline.py                      > out/headline.log 2>&1
$PY report.py
