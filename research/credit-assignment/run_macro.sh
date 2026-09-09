#!/bin/bash
# The composite-action arms of section 8.
cd /home/brandonin/Documents/typed-crystallization-networks
PY=.venv/bin/python
OUT=research/credit-assignment/out
mkdir -p $OUT/logs
pids=()
for s in 0 1 2 3; do
  $PY research/credit-assignment/macro.py 400 $s sweep,stare > $OUT/logs/macro_$s.log 2>&1 &
  pids+=($!)
done
for s in 0 1; do
  $PY research/credit-assignment/macro.py 400 $s stare > $OUT/logs/macro_stare_$s.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
echo "macro done $(date +%T)"
