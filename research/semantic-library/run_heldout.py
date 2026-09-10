"""The held-out protocol: mine from the remainder, evaluate on the task removed.

For each held-out task the library offered to `arm2_syntactic` and
`arm2s_semantic` was mined from a corpus that contains **no program of that
task at any length**.  Every arm is enumerated exhaustively on the compact
scaffold (25 200 programs with a 3-ary module, 4 760 flat) and learned by
gradient over the same 24 seeds.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _paths  # noqa: F401

from tcn.search import enumerate_fit, space_size, candidate_counts

import later

import armlib
import evaltasks

OUT = Path(__file__).resolve().parent / "out"
TASKS = list(evaltasks.LOO_TASKS) + list(evaltasks.CONTROL_TASKS)


def _enum(job):
    task, arm = job
    fn = evaltasks.COMPACT_TASKS[task]
    r, module_name = armlib.build(armlib.heldout_spec(arm, task))
    program = evaltasks.compact_scaffold(r, module_name)
    examples = evaltasks.examples_for(fn)
    counts = candidate_counts(program)
    total = space_size(program)
    t0 = time.perf_counter()
    res = enumerate_fit(program, examples, evaltasks.SIGNALS, registry=r,
                        tolerance=.001, max_programs=1 << 24, rank="order")
    row = res.to_dict()
    row.update(task=task, arm=arm, module=module_name,
               candidates_per_node=list(counts), declared_space_size=total,
               constant_baseline=evaltasks.constant_baseline(fn),
               wall_seconds=time.perf_counter() - t0)
    if res.selections:
        hardened = program.harden(res.selections)
        row["module_on_output_path"] = later.module_on_output_path(hardened)
    return row


def _grad(job):
    task, arm, seed, steps = job
    fn = evaltasks.COMPACT_TASKS[task]
    r, module_name = armlib.build(armlib.heldout_spec(arm, task))
    program = evaltasks.compact_scaffold(r, module_name)
    rec = later.gradient_search(program, evaltasks.examples_for(fn),
                                evaltasks.SIGNALS, r, seed, steps=steps)
    rec.update(task=task, arm=arm, space_size=space_size(program))
    return rec


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    workers = int(os.environ.get("TCN_WORKERS", "16"))
    OUT.mkdir(exist_ok=True)
    if stage in ("all", "enum"):
        jobs = [(t, a) for t in TASKS for a in armlib.HELDOUT_ARMS]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_enum, jobs))
        (OUT / "heldout_enum.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
        for r in rows:
            print(r["task"], r["arm"], "space", r["space_size"], "conforming",
                  r["conforming"], "exhausted", r["exhausted"], r["certificate"],
                  flush=True)
    if stage in ("all", "grad"):
        seeds = 24
        jobs = [(t, a, s, 400) for t in TASKS for a in armlib.HELDOUT_ARMS
                for s in range(seeds)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            recs = list(pool.map(_grad, jobs))
        summary = []
        for t in TASKS:
            for a in armlib.HELDOUT_ARMS:
                rows = [x for x in recs if x["task"] == t and x["arm"] == a]
                hits = [x for x in rows if x["conformant"]]
                steps_to = sorted(x["first_conformant_step"] for x in hits
                                  if x["first_conformant_step"] is not None)
                accs = sorted(x["accuracy"] for x in rows)
                summary.append({
                    "task": t, "arm": a, "seeds": len(rows), "solved": len(hits),
                    "median_steps_to_conformant": (steps_to[len(steps_to) // 2]
                                                   if steps_to else None),
                    "median_accuracy": accs[len(accs) // 2],
                    "mean_accuracy": sum(accs) / len(accs),
                    "best_accuracy": accs[-1],
                    "constant_baseline": evaltasks.constant_baseline(
                        evaltasks.COMPACT_TASKS[t]),
                    "random_baseline": 0.5,
                    "module_on_output_path": sum(1 for x in hits
                                                 if x["module_on_output_path"]),
                })
                print(json.dumps(summary[-1], sort_keys=True), flush=True)
        (OUT / "heldout_gradient.json").write_text(json.dumps(
            {"seeds": 24, "steps": 400, "summary": summary, "records": recs},
            indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
