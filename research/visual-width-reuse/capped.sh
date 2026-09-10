#!/usr/bin/env bash
# Every job in this track runs through here after the 2026-09-10 host crash:
# a user scope with a memory and CPU cap, single-threaded BLAS/OpenMP, and
# /usr/bin/time -v so peak RSS is recorded beside every result.
#   usage: capped.sh <logname> <python args...>
#
# The cap was 40G until the memory probes (out/memprobe_*.time) and the first
# capped arms measured every job in this track at under 1 GB peak RSS, while an
# unrelated process on the host held ~66 GB of unified memory.  Four 40G scopes
# would then have summed past what was free, so the cap was *lowered* to 8G:
# four workers can never claim more than 32G.  Lowered, never raised.
set -euo pipefail
cd "$(dirname "$0")"
log="out/$1.time"; shift
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec systemd-run --user --scope -q -p MemoryMax=8G -p CPUQuota=800% \
  /usr/bin/time -v -o "$log" \
  /home/brandonin/Documents/typed-crystallization-networks/.venv/bin/python -u "$@"
