"""Where the surrogates are on the committed path, and what they read.

The two dead-surrogate faults on record (FINDINGS 16, 19, 21, 23) are properties
of `eq` and of the ordering comparisons at byte distances.  A weighted sum does
not use either.  This prints the surrogate value at this track's operating
distances beside the gradient kind of every operator the convolution path
actually contains, so the claim "the numeric route has no surrogate in it" is a
measurement rather than an assertion.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch

from common import BYTE, FMAG, IDX, MAG8, dump, report
from tcn.learning import relaxed
from tcn.operators import Registry
from tcn.types import product

r = Registry()


def main():
    out = {}

    # 1. the surrogates the byte route depends on, at byte distances
    eq = r.resolve("eq", (BYTE, BYTE))
    le = r.resolve("le", (MAG8, MAG8))
    eqs, les = {}, {}
    for d in (0, 1, 5, 11, 16, 17, 32, 48, 89, 192, 255):
        a = torch.tensor([[0.]], requires_grad=True)
        b = torch.tensor([[float(d)]])
        y = relaxed(r, eq, [a, b], 1.)
        y.sum().backward()
        eqs[d] = {"value": float(y), "grad": float(a.grad)}
        a = torch.tensor([[0.]], requires_grad=True)
        y = relaxed(r, le, [a, b], 1.)
        y.sum().backward()
        les[d] = {"value": float(y), "grad": float(a.grad)}
    out["eq_surrogate_tau1"] = eqs
    out["le_surrogate_tau1"] = les
    print("surrogate value and d/da at tau=1, by operand distance")
    print(f"  {'distance':>9s} {'eq value':>12s} {'eq grad':>12s} {'le value':>12s} {'le grad':>12s}")
    for d in eqs:
        print(f"  {d:>9d} {eqs[d]['value']:>12.4g} {eqs[d]['grad']:>12.4g} "
              f"{les[d]['value']:>12.4g} {les[d]['grad']:>12.4g}")

    # 2. what the committed weighted-sum path is actually made of
    obs = product(*(BYTE for _ in range(9)))
    path = [("add", r.resolve("add", (IDX, IDX))),
            ("index", r.resolve("index", (obs, IDX))),
            ("interpret", r.resolve("interpret", (BYTE,), MAG8)),
            ("decode", r.resolve("decode", (MAG8,), FMAG)),
            ("mul", r.resolve("mul", (FMAG, FMAG))),
            ("sum", r.resolve("sum", (product(FMAG, FMAG, FMAG),)))]
    out["convolution_path"] = {name: op.gradient for name, op in path}
    print()
    print("gradient kind of every operator on the committed weighted-sum path")
    for name, op in path:
        print(f"  {name:<12s} {op.gradient}")
    out["surrogate_free"] = [name for name, op in path if op.gradient == "exact"]
    report("operators on the path with an exact derivative",
           f"{len(out['surrogate_free'])} of {len(path)}: {out['surrogate_free']}")
    report("`eq`/`le` on the path", "none")
    dump("surrogates", out)


if __name__ == "__main__":
    main()
