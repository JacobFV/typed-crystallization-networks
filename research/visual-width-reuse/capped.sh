#!/usr/bin/env bash
# Every job in this track runs through here after the 2026-09-10 host crash:
# a user scope capped at 40 GB and 8 CPUs, single-threaded BLAS/OpenMP, and
# /usr/bin/time -v so peak RSS is recorded beside every result.
#   usage: capped.sh <logname> <python args...>
set -euo pipefail
cd "$(dirname "$0")"
log="out/$1.time"; shift
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec systemd-run --user --scope -q -p MemoryMax=40G -p CPUQuota=800% \
  /usr/bin/time -v -o "$log" \
  /home/brandonin/Documents/typed-crystallization-networks/.venv/bin/python -u "$@"
