#!/bin/bash
# Q2: the differentiable path on the min-prefix scaffold and on the counting-only
# control, at both temperature settings and both step budgets, plus the
# uniform-random-selection null of addendum A2.
#
#   dyck     -- section 45 proves a solution exists in this family
#   counting -- section 45 proves none does (exhausted, 0 conforming, `complete`)
#
# tau_lt 128.0 is the language track's own `stage_b_grad_tau` value; the `in{i}`
# nodes are single-candidate, so raising their temperature widens the `lt`
# surrogate without flattening any choice distribution.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
D=research/dyck-learnability
mkdir -p $D/out
pids=()
for sc in dyck counting; do
  for tau in 0 128; do
    for steps in 600 3000; do
      $PY $D/q2_gradient.py --scaffold $sc --tau-lt $tau --steps $steps --seeds 8 \
          --out $D/out/q2_${sc}_tau${tau}_s${steps}.json \
          > $D/out/q2_${sc}_tau${tau}_s${steps}.log 2>&1 &
      pids+=($!)
    done
  done
done
$PY $D/q2_null.py > $D/out/q2_null.log 2>&1 &
pids+=($!)
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "q2 finished, failures=$fail"
