"""Solve every earlier task and write the corpus of solved programs.

One process per (task, seed); the first conforming seed in seed order is the
solved program kept for that task, so the corpus does not depend on scheduling.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tcn.operators import Registry

from corpus import (EARLIER_TASKS, chain_scaffold, conformant, examples_for,
                    gradient_solve)

OUT = Path(__file__).parent / "out"
SEEDS = tuple(range(8))


def one(args):
    task, seed, depth, steps = args
    fn = dict(EARLIER_TASKS)[task]
    ex = examples_for(fn)
    r = Registry()
    program, signals = chain_scaffold(r, depth=depth)
    model, opt, first, wall = gradient_solve(r and program, ex, signals, r, seed, steps=steps)
    rec = {"task": task, "seed": seed, "depth": depth, "wall_seconds": wall,
           "first_step": first, "conformant": False, "program": None}
    if first is None:
        return rec
    exported = model.export()
    pruned = exported.pruned().validate(r)
    if not conformant(pruned, ex, signals, r):
        return rec
    rec.update(conformant=True, raw_nodes=len(exported.nodes),
               pruned_nodes=len(pruned.nodes), digest=pruned.digest,
               description_bits=pruned.description_bits(),
               execution_cost=pruned.execution_cost(r),
               program=pruned.to_dict())
    return rec


def main():
    OUT.mkdir(exist_ok=True)
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 900
    jobs = [(name, seed, depth, steps) for name, _ in EARLIER_TASKS for seed in SEEDS]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=16) as pool:
        records = list(pool.map(one, jobs))
    programs, report = {}, []
    for name, _ in EARLIER_TASKS:
        rows = [r for r in records if r["task"] == name]
        rows.sort(key=lambda r: r["seed"])
        hit = next((r for r in rows if r["conformant"]), None)
        report.append({"task": name, "solved": hit is not None,
                       "seed": hit["seed"] if hit else None,
                       "first_step": hit["first_step"] if hit else None,
                       "nodes": hit["pruned_nodes"] if hit else None,
                       "digest": hit["digest"] if hit else None,
                       "description_bits": hit["description_bits"] if hit else None,
                       "seeds_conformant": sum(1 for r in rows if r["conformant"]),
                       "seeds_run": len(rows),
                       "wall_seconds": sum(r["wall_seconds"] for r in rows),
                       "steps_to_first": [r["first_step"] for r in rows]})
        if hit:
            programs[name] = hit["program"]
        print(name, "solved" if hit else "FAILED",
              report[-1]["nodes"], report[-1]["seeds_conformant"], "/", len(rows), flush=True)
    (OUT / f"corpus_d{depth}.json").write_text(json.dumps(
        {"depth": depth, "steps": steps, "seeds": list(SEEDS), "programs": programs,
         "report": report, "wall_seconds": time.perf_counter() - t0},
        indent=2, sort_keys=True))
    print("solved", len(programs), "of", len(EARLIER_TASKS),
          "in", round(time.perf_counter() - t0, 1), "s wall")


if __name__ == "__main__":
    main()
