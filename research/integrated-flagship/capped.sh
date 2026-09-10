#!/usr/bin/env bash
# Every heavy job in this track runs through here (resource rules, 2026-09-10):
# a user scope capped at 20G and 800% CPU, single-threaded BLAS/OpenMP, and
# /usr/bin/time -v so peak RSS is recorded beside every result. The memory
# floor (MemAvailable >= 25 GB) is checked before the job starts; below it the
# job is refused and the refusal is logged, never overridden. Never raise the cap.
#   usage: capped.sh <logname> <python args...>
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out
log="out/$1.time"; shift
avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
echo "$(date -Is) $(basename "$log" .time) MemAvailable_kB=$avail_kb" >> out/memory_floor.log
if [ "$avail_kb" -lt $((25 * 1024 * 1024)) ]; then
  echo "$(date -Is) REFUSED $(basename "$log" .time): MemAvailable below 25 GB" >> out/memory_floor.log
  echo "memory floor: MemAvailable ${avail_kb} kB < 25 GB; not starting" >&2
  exit 75
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec systemd-run --user --scope -q -p MemoryMax=20G -p CPUQuota=800% \
  /usr/bin/time -v -o "$log" \
  /home/brandonin/Documents/typed-crystallization-networks/.venv/bin/python -u "$@"
