#!/usr/bin/env bash
# Run arms one after another inside ONE worker slot, each through capped.sh
# (20G cap, memory floor checked before each). Arms are committed by hand as
# each JSON lands, so a crash loses at most one arm.
#   usage: run_queue.sh <gap> <arm> [<arm> ...]      (arm "extras" = H + baselines)
set -uo pipefail
cd "$(dirname "$0")"
gap="$1"; shift
for arm in "$@"; do
  tag="arm_gap${gap}_${arm//\'/p}"
  echo "$(date -Is) start $tag" >> out/queue.log
  ./capped.sh "$tag" arms.py --gap "$gap" --arm "$arm" --samples 400 > "out/$tag.log" 2>&1
  echo "$(date -Is) done $tag exit=$?" >> out/queue.log
done
