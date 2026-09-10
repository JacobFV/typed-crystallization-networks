#!/bin/bash
# Wait for the Q1 shards to land, aggregate them, then run every Q2 arm.
# Q2 is held back until Q1 is done so the enumeration's measured seconds/program
# is not distorted by the gradient arms competing for cores.
set -u
cd "$(dirname "$0")/../.."
D=research/dyck-learnability
PY=.venv/bin/python
while [ "$(ls $D/out/q1_c*.json 2>/dev/null | wc -l)" -lt 11 ] || [ ! -f $D/out/q1b_c95-110.json ]; do
  sleep 30
done
echo "Q1 complete at $(date +%H:%M:%S)"
OMP_NUM_THREADS=1 $PY $D/q1_aggregate.py > $D/out/q1_union.log 2>&1
echo "--- Q1 union ---"
grep -E '"(exhausted|conforming|certificate|evaluated|full_space_size|cpu_hours_total|wall_clock_seconds_parallel|partition_is_exact|index_arithmetic_agrees|reproduces|fraction_through_enumeration|global_enumeration_index)"' $D/out/q1_union.log
echo "--- starting Q2 at $(date +%H:%M:%S) ---"
bash $D/q2_launch.sh
OMP_NUM_THREADS=1 $PY $D/q2_aggregate.py > $D/out/q2_summary.log 2>&1
echo "--- Q2 summary ---"
head -20 $D/out/q2_summary.log
echo "chain finished at $(date +%H:%M:%S)"
