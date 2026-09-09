"""Solve every earlier task exactly, with a minimality certificate.

Each task is solved by iterative-deepening exhaustive search over straight-line
programs in the scaffold basis. Every length below the answer is exhausted, so
the length reported is the certified minimum in that basis; the program
returned is the first at that length in enumeration order. Each circuit is
rebuilt as a `tcn.graph.Program` and executed through `tcn` against the full
truth table before it enters the corpus.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tcn.operators import Registry

import minimal
from corpus import CORPUS_INPUTS, EARLIER_TASKS

OUT = Path(__file__).parent / "out"


def one(name):
    fn = dict(EARLIER_TASKS)[name]
    n = len(CORPUS_INPUTS)
    t0 = time.perf_counter()
    tt = minimal.table_of(n, fn)
    k, gates, stats = minimal.scaffold_min(n, tt, max_len=6)
    wall = time.perf_counter() - t0
    if k is None:
        return {"task": name, "solved": False, "stats": stats, "wall_seconds": wall}
    r = Registry()
    program = minimal.to_program(n, gates, CORPUS_INPUTS, r)
    ok = minimal.verify(program, fn, CORPUS_INPUTS, r)
    return {"task": name, "solved": bool(ok), "min_gates": k,
            "certificate": f"minimum in basis; lengths {stats['exhausted_lengths']} exhausted",
            "exhausted_lengths": stats["exhausted_lengths"],
            "nodes_expanded": stats["nodes_expanded"],
            "total_expanded": sum(s["expanded"] for s in stats["nodes_expanded"]),
            "gates": [[g[0], g[1], g[2]] for g in gates],
            "digest": program.digest, "nodes": len(program.nodes),
            "description_bits": program.description_bits(),
            "execution_cost": program.execution_cost(r),
            "verified_in_tcn": bool(ok), "wall_seconds": wall,
            "program": program.to_dict()}


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=6) as pool:
        recs = list(pool.map(one, [n for n, _ in EARLIER_TASKS]))
    programs = {r["task"]: r["program"] for r in recs if r.get("solved")}
    for r in recs:
        print(r["task"], r.get("min_gates"), "gates", "verified" if r.get("solved") else "FAILED",
              [g[0] for g in r.get("gates", [])], round(r["wall_seconds"], 1), "s", flush=True)
    (OUT / "corpus_exact.json").write_text(json.dumps(
        {"solver": "exhaustive straight-line search in the scaffold basis",
         "programs": programs,
         "report": [{k: v for k, v in r.items() if k != "program"} for r in recs],
         "wall_seconds": time.perf_counter() - t0}, indent=2, sort_keys=True))
    print("solved", len(programs), "of", len(EARLIER_TASKS))


if __name__ == "__main__":
    main()
