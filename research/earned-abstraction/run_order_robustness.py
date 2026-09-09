"""How much does the corpus -- and so the proposal -- depend on the solver's
arbitrary tie-break among equally minimal programs?

Each task has many minimum-length straight-line programs; the exact solver
returns the first in *its* enumeration order, exactly as
`tcn.search.enumerate_fit(rank='order')` does. This sweeps the six permutations
of the binary operator order, re-solves the whole corpus under each, and
re-runs the rule. Every permutation's outcome is reported; none is selected.
"""
from __future__ import annotations

import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tcn.operators import Registry

import mine
import minimal
from corpus import CORPUS_INPUTS, EARLIER_TASKS, examples_for, maj

OUT = Path(__file__).parent / "out"
BASE = dict(minimal.OPS)
MAJ_TT = tuple((b, maj(*b)) for b in itertools.product((False, True), repeat=3))


def solve(job):
    name, order = job
    minimal.OPS = tuple((k, BASE[k]) for k in order)
    fn = dict(EARLIER_TASKS)[name]
    n = len(CORPUS_INPUTS)
    k, gates, stats = minimal.scaffold_min(n, minimal.table_of(n, fn), max_len=6)
    r = Registry()
    program = minimal.to_program(n, gates, CORPUS_INPUTS, r)
    ok = minimal.verify(program, fn, CORPUS_INPUTS, r)
    return {"order": list(order), "task": name, "min_gates": k, "verified": bool(ok),
            "gates": [[g[0], g[1], g[2]] for g in gates], "program": program.to_dict()}


def main():
    orders = list(itertools.permutations(("and", "or", "xor")))
    jobs = [(name, order) for order in orders for name, _ in EARLIER_TASKS]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=12) as pool:
        recs = list(pool.map(solve, jobs))
    fns = dict(EARLIER_TASKS)
    out = []
    for order in orders:
        rows = [r for r in recs if tuple(r["order"]) == order]
        corpus = {}
        for row in rows:
            from tcn.graph import Program
            corpus[row["task"]] = Program.from_dict(row["program"], Registry())
        ex = {k: examples_for(fns[k]) for k in corpus}
        props = mine.propose(corpus, ex)
        top = props[0] if props else None
        maj_bodies = 0
        for task, p in corpus.items():
            if any(len(c.inputs) == 3 and mine.truth_table(c) == MAJ_TT
                   for _, _, c, _ in mine.fragments(p)):
                maj_bodies += 1
        maj_prop = next((q for q in props if q.holes == 3
                         and mine.truth_table(q.canonical) == MAJ_TT), None)
        out.append({
            "order": list(order),
            "min_gates": {r["task"]: r["min_gates"] for r in rows},
            "all_verified": all(r["verified"] for r in rows),
            "tasks_containing_a_maj3_body": maj_bodies,
            "top_digest": top.digest if top else None,
            "top_nodes": top.nodes if top else None,
            "top_holes": top.holes if top else None,
            "top_saving_bits": top.saving_bits if top else None,
            "top_is_maj3": bool(top and top.holes == 3
                                and mine.truth_table(top.canonical) == MAJ_TT),
            "maj3_rank": maj_prop.rank if maj_prop else None,
            "maj3_saving_bits": maj_prop.saving_bits if maj_prop else None,
        })
        print(json.dumps(out[-1], sort_keys=True), flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "order_robustness.json").write_text(json.dumps(
        {"orders": out, "wall_seconds": time.perf_counter() - t0}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
