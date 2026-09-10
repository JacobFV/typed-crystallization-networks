#!/bin/bash
# Q1: exhaust the FULL min-prefix scaffold space (680,625 programs) as eleven
# disjoint, contiguous, order-preserving shards of the `c` axis, plus arm 1b,
# the reproduction control over section 45's hand-chosen window c in [95,110).
#
# The search is `tcn.search.enumerate_fit`, unmodified. `sub_range` is an
# argument section 45's own runner already exposes. The shards partition
# [0,121) exactly, so eleven `exhausted` shards decide the whole space.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
D=research/dyck-learnability
mkdir -p $D/out
pids=()
for lo in 0 11 22 33 44 55 66 77 88 99 110; do
  hi=$((lo + 11))
  $PY $D/q1_enum.py --sub-lo $lo --sub-hi $hi > $D/out/q1_c${lo}-${hi}.log 2>&1 &
  pids+=($!)
done
# arm 1b: the reproduction control on section 45's window
$PY $D/q1_enum.py --sub-lo 95 --sub-hi 110 --rank description \
    --out $D/out/q1b_c95-110.json > $D/out/q1b_c95-110.log 2>&1 &
pids+=($!)
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "shards finished, failures=$fail"
