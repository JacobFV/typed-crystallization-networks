"""What would make exhaustive search affordable: reuse the prefix.

`tcn.search.enumerate_fit` re-executes the *whole* program for every discrete
selection.  In a per-position module almost every node is fixed -- the three
`index` reads that decode the observation do not depend on any choice -- so the
same work is repeated once per candidate program.  At the shipped
`Value.decoded`, one such re-execution walks the entire byte tuple, which is why
a 32,000-program sweep over a few hundred records costs minutes rather than
seconds and why the cost grows with image width even though the space does not.

This is the same search, in the same order, over the same candidates, evaluated
as a DFS over the choice tree with each node's value computed once per prefix
rather than once per leaf.  It returns the identical conforming set, which is
checked here against `common.all_conforming`.

Nothing under `tcn/` is modified: this is a prototype for the diff proposed in
RESULTS.md section 12.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import all_conforming, dump, report
from rung3_mask import POOL, module_scaffold, pixel_examples, signals
from tcn.operators import Registry
from tcn.search import enumerate_fit, space_size


def incremental_conforming(program, examples, signals_, registry, tolerance=1e-6,
                           early_exit=True):
    """Every conforming selection, with node values reused across the prefix."""
    program.validate(registry)
    program.validate_signals(signals_)
    nodes = list(program.nodes)
    n = len(examples)
    base = []
    for ex in examples:
        v = dict(ex["inputs"]) | dict(program.constants)
        base.append(v)
    targets = {s.target: [ex["targets"][s.target].flat() for ex in examples] for s in signals_}
    by_source = {}
    for s in signals_:
        by_source.setdefault(s.source, []).append(s)
    values = [dict(v) for v in base]
    found, leaves, node_evals = [], 0, 0
    sel = {}

    def ok(name):
        for s in by_source[name]:
            tgt = targets[s.target]
            for e in range(n):
                a = values[e][name].flat()
                b = tgt[e]
                for x, y in zip(a, b):
                    if abs(x - y) > tolerance:
                        return False
        return True

    def rec(i):
        nonlocal leaves, node_evals
        if i == len(nodes):
            leaves += 1
            found.append(dict(sel))
            return
        node = nodes[i]
        probed = node.name in by_source
        for k, c in enumerate(node.candidates):
            try:
                if probed and early_exit:
                    # A probed node is checked example by example, so a candidate
                    # that already disagrees costs one example rather than all of
                    # them -- the same short-circuit `tcn.search.evaluate` uses.
                    bad = False
                    for e in range(n):
                        v = registry.exact(c.operator, [values[e][s] for s in c.sources])
                        values[e][node.name] = v
                        a = v.flat()
                        for s in by_source[node.name]:
                            for x, y in zip(a, targets[s.target][e]):
                                if abs(x - y) > tolerance:
                                    bad = True; break
                            if bad: break
                        if bad: break
                    node_evals += 1
                    if bad:
                        continue
                else:
                    for e in range(n):
                        values[e][node.name] = registry.exact(
                            c.operator, [values[e][s] for s in c.sources])
                    node_evals += 1
            except (ValueError, TypeError, IndexError, OverflowError, ZeroDivisionError,
                    ArithmeticError):
                continue
            sel[node.name] = k
            if probed and not early_exit and not ok(node.name):
                continue
            rec(i + 1)
        sel.pop(node.name, None)

    t0 = time.perf_counter()
    rec(0)
    return {"conforming": found, "count": len(found), "node_evaluations": node_evals,
            "leaves_reached": leaves, "space_size": space_size(program),
            "seconds": time.perf_counter() - t0, "exhausted": True,
            "unique": len(found) == 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolutions", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--per-image", type=int, default=48)
    ap.add_argument("--objects", type=int, default=6)
    ap.add_argument("--baseline-resolution", type=int, default=8)
    ap.add_argument("--tag", default="incremental")
    args = ap.parse_args()

    tol = 1e-6
    out = {"arguments": vars(args), "rows": []}
    for R in args.resolutions:
        r = Registry()
        train = pixel_examples(tuple(range(args.train)), R, "train", args.per_image, seed=1,
                              objects=args.objects)
        prog = module_scaffold(r, R, POOL)
        sig = signals()
        fast = incremental_conforming(prog, train, sig, r, tolerance=tol)
        row = {"resolution": R, "observation_width": 3 * R * R, "records": len(train),
               "space_size": fast["space_size"], "incremental": {
                   k: fast[k] for k in ("count", "node_evaluations", "leaves_reached",
                                        "seconds", "exhausted", "unique")}}
        report(f"R={R:3d} incremental: conforming {fast['count']} in {fast['seconds']:.2f}s "
               f"({fast['node_evaluations']:,} node evaluations)", "")
        if R == args.baseline_resolution:
            slow = all_conforming(prog, train, sig, r, tolerance=tol)
            row["enumerate_fit_equivalent"] = {k: slow[k] for k in
                                               ("count", "evaluated", "seconds", "exhausted",
                                                "unique")}
            row["identical_conforming_set"] = (
                sorted(map(sorted, (s.items() for s in fast["conforming"]))) ==
                sorted(map(sorted, (s.items() for s in slow["conforming"]))))
            row["speedup"] = slow["seconds"] / max(1e-9, fast["seconds"])
            report(f"R={R} exhaustive baseline {slow['seconds']:.1f}s -> incremental "
                   f"{fast['seconds']:.2f}s", f"{row['speedup']:.0f}x, identical set: "
                   f"{row['identical_conforming_set']}")
        out["rows"].append(row)
        dump(args.tag, out)


if __name__ == "__main__":
    main()
