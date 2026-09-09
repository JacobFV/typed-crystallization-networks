"""The discrete reference: does exhaustive enumeration select the module?

`tcn/search.py` enumerates exactly the space `SoftProgram` relaxes, using only
`Program.execute` over the declared node candidates. Running it on the same
scaffold the gradient search gets separates two very different failures:

  * enumeration finds no module solution  -> the abstraction does not pay, and
    no search method could have shown otherwise;
  * enumeration finds it and the gradient search does not -> the accounting is
    fine and the relaxation is the problem.

The tight scaffold (3 nodes) is the decisive case: its 3 nodes are below the
verified flat minimum of 7, so a solution exists in arm B and cannot exist in
arm A. Enumerating arm A therefore also certifies, exhaustively, that the
abstraction is not merely convenient but *necessary* at this scaffold size.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tcn.operators import Registry
from tcn.graph import Program
from tcn.search import enumerate_fit, space_size

import common as C

HERE = Path(__file__).parent


def describe(program, selections, registry):
    if selections is None:
        return None
    hard = program.harden(selections).validate(registry)
    p = C.prune(hard)
    return {
        "selections": {n.name: {"operator": n.candidates[n.selected or 0].operator.name[:20],
                                "sources": list(n.candidates[n.selected or 0].sources)}
                       for n in p.nodes},
        "live_nodes": len(p.nodes),
        "module_on_output_path": C.module_on_output_path(hard),
        "description_bits_pruned": p.description_bits(registry),
        "description_bits_pruned_no_library": p.description_bits(),
        "execution_cost_pruned": p.execution_cost(registry),
    }


def run(arm, module_fn, max_programs, stop_at_first):
    r = Registry()
    name = None
    if module_fn is not None:
        body = C.minimal_module(r, "maj" if module_fn is C.maj else "distractor")
        name = r.register_module(body)
        assert C.verify_module(body, module_fn, r)
    prog = C.tight_scaffold(r, name)
    ex = C.composite_examples()
    total = space_size(prog)
    t = time.perf_counter()
    res = enumerate_fit(prog, ex, C.COMP_SIGNALS, registry=r, tolerance=0.001,
                        max_programs=max_programs, stop_at_first=stop_at_first)
    d = res.to_dict()
    d.update(arm=arm, declared_space_size=total, wall_seconds=round(time.perf_counter() - t, 1),
             program=describe(prog, res.selections, r),
             candidates_per_node=[len(n.candidates) for n in prog.nodes])
    d.pop("selections", None)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-programs", type=int, default=4_000_000)
    ap.add_argument("--arms", default="A,B")
    ap.add_argument("--stop-at-first", action="store_true")
    ap.add_argument("--out", default="enumeration.json")
    args = ap.parse_args()

    out = {"config": vars(args), "results": []}
    for arm in args.arms.split(","):
        fn = {"A": None, "B": C.maj, "C": C.distractor}[arm]
        d = run(arm, fn, args.max_programs, args.stop_at_first)
        out["results"].append(d)
        print(json.dumps(d, indent=2), flush=True)
        Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    out["independent_solution_count"] = solution_count()
    print(json.dumps(out["independent_solution_count"], indent=2), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
def solution_count():
    """Count the tight scaffold's solutions independently of `tcn.search`.

    `enumerate_fit` reports `unique = False` but not how many. Recomputing the
    count from packed truth tables -- no tcn execution at all -- both quantifies
    the degeneracy and cross-checks the enumerator against a second method.
    """
    import itertools
    rows = list(itertools.product((0, 1), repeat=6))
    col = {k: sum(r[i] << n for n, r in enumerate(rows)) for i, k in enumerate(C.INPUTS)}
    m = (1 << 64) - 1
    target = sum(int(C.composite([bool(x) for x in r])) << n for n, r in enumerate(rows))

    def maj_tab(x, y, z):
        return (x & y) | (z & (x | y))

    # every module binding at n1/n2, in the scaffold's own order
    mod = {t: maj_tab(col[t[0]], col[t[1]], col[t[2]])
           for t in itertools.product(C.INPUTS, repeat=3)}
    prim = {}
    for op, fn in (("and", lambda u, v: u & v), ("or", lambda u, v: u | v), ("xor", lambda u, v: u ^ v)):
        for p, q in itertools.product(C.INPUTS, repeat=2):
            prim[(op, p, q)] = fn(col[p], col[q])
    for p in C.INPUTS:
        prim[("not", p, p)] = (~col[p]) & m
        prim[("identity", p, p)] = col[p]

    n1_vals = list(prim.values()) + list(mod.values())
    total = 0
    module_solutions = 0
    n_prim = len(prim)
    for i, u in enumerate(n1_vals):
        for j, v in enumerate(n1_vals):
            # y candidates over (n1, n2): three binary ops on ordered pairs,
            # not/identity on each, plus module calls over the two-name pool
            ys = []
            for a, b in ((u, u), (u, v), (v, u), (v, v)):
                ys += [a & b, a | b, a ^ b]
            ys += [(~u) & m, (~v) & m, u, v]
            ys += [maj_tab(*t) for t in itertools.product((u, v), repeat=3)]
            hits = sum(1 for w in ys if w == target)
            total += hits
            if hits and (i >= n_prim or j >= n_prim):
                module_solutions += hits
    return {"total_solutions": total, "solutions_using_a_module": module_solutions,
            "space_size": len(n1_vals) ** 2 * 24,
            "method": "packed truth tables, independent of tcn.search"}
