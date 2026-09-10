"""Gradient conformance beside the enumeration, on the same held-out protocol.

`later.gradient_search` unchanged, 24 seeds, 400 steps, exactly as §52.  The
enumeration is the decisive measurement -- it carries a certificate -- and this
is reported beside it because §52 reported it and the two should be compared on
the same rows.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _paths  # noqa: F401

from tcn.search import space_size

import evaltasks
import later
import objectives
import run_heldout as H

OUT = Path(__file__).resolve().parent / "out"
SEEDS = 24
STEPS = 400


def _grad(job):
    task, spec, seed = job
    fn = evaltasks.COMPACT_TASKS[task]
    r, module_name = H.registry_for(spec)
    program = evaltasks.compact_scaffold(r, module_name)
    rec = later.gradient_search(program, evaltasks.examples_for(fn),
                                evaltasks.SIGNALS, r, seed, steps=STEPS)
    rec.update(task=task, spec=spec, space_size=space_size(program))
    return rec


def main():
    ranked = json.loads((OUT / "ranked.json").read_text())
    specs = {"none", "hand:maj", "hand:distractor"}
    for t in H.LOO_TASKS:
        cname = "wo_" + t.split("_")[0]
        for oid in objectives.ALL_IDS:
            specs.add("lib:" + ranked["corpora"][cname]["tables"][oid]["rank1"]["library_label"])
    jobs = [(t, s, seed) for t in H.LOO_TASKS + H.CONTROLS
            for s in sorted(specs) for seed in range(SEEDS)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(os.environ.get("TCN_WORKERS", "16"))) as ex:
        recs = list(ex.map(_grad, jobs, chunksize=4))
    summary = []
    for t in H.LOO_TASKS + H.CONTROLS:
        for s in sorted(specs):
            rows = [x for x in recs if x["task"] == t and x["spec"] == s]
            hits = [x for x in rows if x["conformant"]]
            accs = sorted(x["accuracy"] for x in rows)
            summary.append({"task": t, "spec": s, "seeds": len(rows), "solved": len(hits),
                            "median_accuracy": accs[len(accs) // 2],
                            "best_accuracy": accs[-1],
                            "constant_baseline": evaltasks.constant_baseline(
                                evaltasks.COMPACT_TASKS[t]),
                            "random_baseline": 0.5})
            print(json.dumps(summary[-1], sort_keys=True), flush=True)
    (OUT / "heldout_gradient.json").write_text(json.dumps(
        {"seeds": SEEDS, "steps": STEPS, "summary": summary, "records": recs,
         "wall_seconds": time.perf_counter() - t0}, indent=1, sort_keys=True))
    print("->", OUT / "heldout_gradient.json")


if __name__ == "__main__":
    main()
