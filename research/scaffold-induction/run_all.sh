#!/usr/bin/env bash
# Reproduce the whole track, in order, under the resource caps.
#
#   ./run_all.sh
#
# Every heavy phase runs inside a capped transient scope (MemoryMax 20G,
# CPUQuota 100% per worker, single-threaded BLAS) and refuses to start below the
# 25 GB MemAvailable floor, which `kit.check_floor` logs to out/memory_floor.log.
# At most four workers run at once.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
PY="$ROOT/.venv/bin/python"
CAP=(systemd-run --user --scope -q -p MemoryMax=20G -p CPUQuota=100%
     env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1)

# --- corpora: three domains, four workers (arith is sharded by defect index)
"${CAP[@]}" "$PY" run_domain.py bool --max-cases 8            > out/bool.log     2>&1 &
"${CAP[@]}" "$PY" run_domain.py rel  --max-cases 8            > out/rel.log      2>&1 &
"${CAP[@]}" "$PY" run_domain.py arith --shard 0 --of 2 --max-cases 4 > out/arith_s0.log 2>&1 &
"${CAP[@]}" "$PY" run_domain.py arith --shard 1 --of 2 --max-cases 4 > out/arith_s1.log 2>&1 &
wait

"$PY" merge_shards.py arith

# --- the pre-registered secondary arm S1 needs section 50's split
( cd ../scaffold-diagnosis && "${CAP[@]}" "$PY" prepare.py ) > out/prepare.log 2>&1
"${CAP[@]}" "$PY" s50_rescore.py > out/s50.log 2>&1

# --- the arms, the report, the verifier
"$PY" analyse.py
"$PY" report.py --fill
"$PY" verify.py
