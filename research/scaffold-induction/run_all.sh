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

# --- the episode budgets, fixed by the rule of amendment A8 before any corpus
"${CAP[@]}" "$PY" episode_sweep.py bool rel arith > out/sweep.log 2>&1
# --- the exhaustive admissible-defect ceiling per domain
"${CAP[@]}" "$PY" admissible.py bool rel arith > out/admissible.log 2>&1

# --- corpora, at most four workers.  bool is one job (11 admissible); rel and
# arith are sharded by defect index.  Case counts are the domain ceilings.
"${CAP[@]}" "$PY" run_domain.py bool --max-cases 11                  > out/bool.log    2>&1 &
"${CAP[@]}" "$PY" run_domain.py rel  --shard 0 --of 2 --max-cases 10 > out/rel_s0.log  2>&1 &
"${CAP[@]}" "$PY" run_domain.py rel  --shard 1 --of 2 --max-cases 10 > out/rel_s1.log  2>&1 &
wait
for i in 0 1 2 3; do
  "${CAP[@]}" "$PY" run_domain.py arith --shard $i --of 4 --max-cases 3 \
      > out/arith_s$i.log 2>&1 &
done
wait

"$PY" merge_shards.py rel arith

# --- the pre-registered secondary arm S1 needs section 50's split
( cd ../scaffold-diagnosis && "${CAP[@]}" "$PY" prepare.py ) > out/prepare.log 2>&1
"${CAP[@]}" "$PY" s50_rescore.py > out/s50.log 2>&1

# --- V2, the decider's own cross-checks, one domain at a time
for d in bool rel arith; do
  "${CAP[@]}" "$PY" validate_decider.py "$d" --per-case 40 > "out/validate_$d.log" 2>&1
done

# --- the arms, the report, the verifier
"$PY" analyse.py
"$PY" report.py --fill
"$PY" verify.py
