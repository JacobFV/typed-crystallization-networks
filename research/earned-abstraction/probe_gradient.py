"""Can the shipped gradient path solve the earlier tasks at all, given more budget?"""
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


def one(job):
    task, seed, depth, steps = job
    fn = dict(EARLIER_TASKS)[task]
    ex = examples_for(fn)
    r = Registry()
    program, signals = chain_scaffold(r, depth=depth)
    model, opt, first, wall = gradient_solve(program, ex, signals, r, seed, steps=steps)
    ok = False
    nodes = None
    if first is not None:
        pruned = model.export().pruned().validate(r)
        ok = conformant(pruned, ex, signals, r)
        nodes = len(pruned.nodes)
    return {"task": task, "seed": seed, "depth": depth, "steps": steps,
            "first_step": first, "conformant": bool(ok), "pruned_nodes": nodes,
            "wall_seconds": wall}


def main():
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    jobs = [(name, s, depth, steps) for name, _ in EARLIER_TASKS for s in range(seeds)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=18) as pool:
        recs = list(pool.map(one, jobs))
    for name, _ in EARLIER_TASKS:
        rows = [r for r in recs if r["task"] == name]
        print(name, sum(1 for r in rows if r["conformant"]), "/", len(rows), flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / f"probe_gradient_d{depth}_s{steps}.json").write_text(
        json.dumps({"records": recs, "wall_seconds": time.perf_counter() - t0},
                   indent=2, sort_keys=True))
    print("wall", round(time.perf_counter() - t0, 1))


if __name__ == "__main__":
    main()
