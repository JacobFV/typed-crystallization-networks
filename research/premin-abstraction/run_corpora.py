"""Build every declared corpus variant, verify it in `tcn`, and measure retention.

One process per earlier task.  Writes `out/corpora.json`: per task and per
length band the method (`exhaustive` or `sampled`), the exhaustion flag, the
uncapped count, the DFS nodes expanded, the wall clock, the capped programs
themselves, and the retention counts over the *full* enumerated set.

Only the capped programs are stored, so the artifact stays small; the counts
and retention figures are computed over everything enumerated.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tcn.operators import Registry

import corpora
import minimal
import retention
from corpus import CORPUS_INPUTS, EARLIER_TASKS

OUT = Path(__file__).parent / "out"
BUDGET_SECONDS = 1800
TRIALS = 400_000
SEED = 20260909


def retention_block(gate_lists):
    out = {}
    for label, fn in (("MAJ3", retention.maj3), ("M", retention.frag_M),
                      ("Mprime", retention.frag_Mprime)):
        hit, total, per_triple = retention.value_retention(gate_lists, fn)
        out[label] = {"programs_retaining": hit, "programs": total,
                      "fraction": (hit / total) if total else 0.0,
                      "per_ordered_triple": per_triple}
    return out


def one(name):
    fn = dict(EARLIER_TASKS)[name]
    k = corpora.min_lengths()[name]
    cmin_key = corpora._hash(corpora.c_min()[name])
    rec = {"task": name, "min_gates": k, "bands": {}}
    for label, length in (("min", k), ("plus1", k + 1)):
        band = corpora.enumerate_band(fn, length, BUDGET_SECONDS, TRIALS, SEED)
        programs = band.pop("programs")
        band["uncapped_found"] = len(programs)
        band["contains_c_min_program"] = cmin_key in {corpora._hash(p) for p in programs}
        band["retention_full"] = retention_block(programs)
        kept = corpora.subsample(programs)
        band["capped"] = len(kept)
        band["retention_capped"] = retention_block(kept)
        t0 = time.perf_counter()
        band["verified_in_tcn"] = corpora.verify_all(kept, fn)
        band["verify_seconds"] = time.perf_counter() - t0
        band["programs"] = [[list(g) for g in gates] for gates in kept]
        band["first_in_order"] = [list(g) for g in programs[0]] if programs else None
        rec["bands"][label] = band
    return rec


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    names = [n for n, _ in EARLIER_TASKS]
    with ProcessPoolExecutor(max_workers=6) as pool:
        recs = list(pool.map(one, names))

    cmin = corpora.c_min()
    for rec in recs:
        print(rec["task"], "k=", rec["min_gates"],
              {b: (rec["bands"][b]["method"], rec["bands"][b]["uncapped_found"],
                   round(rec["bands"][b]["wall_seconds"], 1)) for b in rec["bands"]},
              flush=True)

    payload = {"cap": corpora.CAP, "budget_seconds": BUDGET_SECONDS,
               "sampler_trials": TRIALS, "sampler_seed": SEED,
               "inputs": list(CORPUS_INPUTS), "tasks": recs,
               "c_min": {k: [list(g) for g in v] for k, v in cmin.items()},
               "wall_seconds": time.perf_counter() - t0}
    (OUT / "corpora.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print("total wall", round(payload["wall_seconds"], 1), "s")


if __name__ == "__main__":
    main()
