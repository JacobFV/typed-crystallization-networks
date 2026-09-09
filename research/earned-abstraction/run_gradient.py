"""Sample complexity: the same later task, learned rather than enumerated.

Every arm gets the same seeds, the same schedule and the same objective; the
only difference between arms is the library. Reports optimizer steps to first
conforming export, per seed, plus the success rate and the exact accuracy of
the export against the constant/random baseline of 0.5.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tcn.search import space_size, candidate_counts

import arms
import later

OUT = Path(__file__).parent / "out"


def one(job):
    arm, scaffold, seed, steps = job
    r, module_name = arms.build(arm)
    program = (later.tight_scaffold(r, module_name) if scaffold == "tight"
               else later.wide_scaffold(r, module_name))
    rec = later.gradient_search(program, later.later_examples(), later.SIGNALS, r,
                                seed, steps=steps)
    rec.update(arm=arm, scaffold=scaffold,
               candidates_per_node=list(candidate_counts(program)),
               space_size=space_size(program))
    return rec


def main():
    scaffold = sys.argv[1] if len(sys.argv) > 1 else "tight"
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    jobs = [(arm, scaffold, s, steps) for arm in arms.ARMS for s in range(seeds)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(os.environ.get("TCN_WORKERS", "16"))) as pool:
        records = list(pool.map(one, jobs))
    summary = []
    for arm in arms.ARMS:
        rows = [r for r in records if r["arm"] == arm]
        hits = [r for r in rows if r["conformant"]]
        steps_to = sorted(r["first_conformant_step"] for r in hits
                          if r["first_conformant_step"] is not None)
        summary.append({
            "arm": arm, "description": arms.DESCRIPTION[arm], "scaffold": scaffold,
            "seeds": len(rows), "solved": len(hits),
            "space_size": rows[0]["space_size"],
            "candidates_per_node": rows[0]["candidates_per_node"],
            "median_steps_to_conformant": (steps_to[len(steps_to) // 2] if steps_to else None),
            "mean_steps_to_conformant": (sum(steps_to) / len(steps_to) if steps_to else None),
            "steps_to_conformant": steps_to,
            "mean_accuracy": sum(r["accuracy"] for r in rows) / len(rows),
            "best_accuracy": max(r["accuracy"] for r in rows),
            "module_on_output_path": sum(1 for r in hits if r["module_on_output_path"]),
            "mean_seconds_per_step": sum(r["seconds_per_step"] for r in rows) / len(rows),
            "total_wall_seconds": sum(r["wall_seconds"] for r in rows),
        })
        print(json.dumps(summary[-1], sort_keys=True), flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / f"gradient_{scaffold}.json").write_text(json.dumps(
        {"scaffold": scaffold, "seeds": seeds, "steps": steps,
         "constant_baseline_accuracy": 0.5, "summary": summary, "records": records,
         "wall_seconds": time.perf_counter() - t0}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
