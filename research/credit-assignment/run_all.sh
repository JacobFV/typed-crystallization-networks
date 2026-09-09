#!/bin/bash
# Every learning arm, in waves. Concurrency is deliberately low: the checkout is
# shared with other agents and the machine already runs at load ~50 on 20 cores.
# Each process owns one `engine/session.ts` and one seed.
cd /home/brandonin/Documents/typed-crystallization-networks
PY=.venv/bin/python
OUT=research/credit-assignment/out
mkdir -p $OUT/logs

wave() {
  local pids=()
  for spec in "$@"; do
    set -- $spec
    $PY research/credit-assignment/arms.py $1 $2 $3 > $OUT/logs/$1_s$2.log 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
}

echo "wave 1 $(date +%T)"
wave "reward 0 600" "reward 1 600" "reward 2 600" \
     "reward 3 600" "reward 4 600" "reward 5 600"
echo "wave 2 $(date +%T)"
wave "myopic 0 600" "myopic 1 600" "myopic 2 600" "myopic 3 600" \
     "flat 0 600" "flat 1 600"
echo "wave 3 $(date +%T)"
wave "flat 2 600" "reward_g50 0 600" "reward_g50 1 600" "reward_g50 2 600" \
     "probe_only 0 300" "probe_only 1 300"
echo "wave 4 $(date +%T)"
wave "staged_enum 0 600" "staged_enum 1 600" "staged_enum 2 600" \
     "reward_percept 0 600" "reward_percept 1 600" "reward_percept 2 600"
echo "done $(date +%T)"
