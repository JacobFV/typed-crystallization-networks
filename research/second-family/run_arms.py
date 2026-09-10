"""§52's arm table on the second family's seventh task.

`L1_w4_bdae_xor_c` is in no mining corpus at any length and uses a pairing no
corpus task uses, so an arm that solves it has transferred the window rather
than memorised a task.  Every arm is enumerated exhaustively on the two-node
compact scaffold with `max_programs` far above the space size, so every row
carries `evaluated == space_size` and a certificate, and every arm is also run
by gradient over the same seeds.

The two off-family controls run in the same panel: no `W4`-family arm, the
hand-authored ceiling included, may score above 0 on either.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _paths  # noqa: F401

from tcn.search import candidate_counts, enumerate_fit, space_size

import later

import armlib
import evaltasks

OUT = Path(__file__).resolve().parent / "out"

ARMS = ("arm1_none", "arm2_syntactic", "arm2s_semantic", "arm3_authored",
        "arm4_wrong_authored", "arm4b_wrong_mined", "arm4s_runnerup",
        "arm4s_matched", "arm4s_offfamily")
TASKS = (evaltasks.LATER_TASK, "H_par5", "H_x4")


def _enum(job):
    band, arm, task = job
    armlib.use_band(band)
    fn = evaltasks.COMPACT_TASKS[task]
    r, module_name = armlib.build(armlib.L1_ARMS[arm])
    program = evaltasks.compact_scaffold(r, module_name)
    examples = evaltasks.examples_for(fn)
    t0 = time.perf_counter()
    res = enumerate_fit(program, examples, evaltasks.SIGNALS, registry=r,
                        tolerance=.001, max_programs=1 << 24, rank="order")
    row = res.to_dict()
    row.update(band=band, arm=arm, task=task, module=module_name,
               description=armlib.DESCRIPTION[arm],
               candidates_per_node=list(candidate_counts(program)),
               declared_space_size=space_size(program),
               constant_baseline=evaltasks.constant_baseline(fn),
               random_baseline=0.5, wall_seconds=time.perf_counter() - t0)
    if res.selections:
        hardened = program.harden(res.selections)
        row["module_on_output_path"] = later.module_on_output_path(hardened)
        row["live_nodes"] = len(hardened.pruned().nodes)
    return row


def _grad(job):
    band, arm, task, seed, steps = job
    armlib.use_band(band)
    fn = evaltasks.COMPACT_TASKS[task]
    r, module_name = armlib.build(armlib.L1_ARMS[arm])
    program = evaltasks.compact_scaffold(r, module_name)
    rec = later.gradient_search(program, evaltasks.examples_for(fn),
                                evaltasks.SIGNALS, r, seed, steps=steps)
    rec.update(band=band, arm=arm, task=task, space_size=space_size(program))
    return rec


def main(stage="all", bands=("C-trace", "C-minall"), seeds=24, steps=400):
    OUT.mkdir(exist_ok=True)
    workers = int(os.environ.get("TCN_WORKERS", "16"))
    if stage in ("all", "enum"):
        jobs = [(b, a, t) for b in bands for a in ARMS for t in TASKS]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_enum, jobs))
        (OUT / "arms_enum.json").write_text(json.dumps(rows, indent=1, sort_keys=True))
        for r in rows:
            print(f"{r['band']:9s} {r['arm']:22s} {r['task']:18s} space "
                  f"{r['space_size']:8d} evaluated {r['evaluated']:8d} conforming "
                  f"{r['conforming']:5d} exhausted {r['exhausted']} "
                  f"{r['certificate']}", flush=True)
    if stage in ("all", "grad"):
        jobs = [(b, a, t, s, steps) for b in bands for a in ARMS for t in TASKS
                for s in range(seeds)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            recs = list(pool.map(_grad, jobs))
        summary = []
        for b in bands:
            for a in ARMS:
                for t in TASKS:
                    rows = [x for x in recs if x["band"] == b and x["arm"] == a
                            and x["task"] == t]
                    hits = [x for x in rows if x["conformant"]]
                    accs = sorted(x["accuracy"] for x in rows)
                    summary.append({
                        "band": b, "arm": a, "task": t, "seeds": len(rows),
                        "solved": len(hits),
                        "median_accuracy": accs[len(accs) // 2],
                        "mean_accuracy": sum(accs) / len(accs),
                        "best_accuracy": accs[-1],
                        "constant_baseline": evaltasks.constant_baseline(
                            evaltasks.COMPACT_TASKS[t]),
                        "random_baseline": 0.5})
                    print(json.dumps(summary[-1], sort_keys=True), flush=True)
        (OUT / "arms_gradient.json").write_text(json.dumps(
            {"seeds": seeds, "steps": steps, "summary": summary, "records": recs},
            indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
