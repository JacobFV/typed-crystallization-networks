"""Enumeration's reach on the 3x3 kernel, measured rather than asserted.

Arm A of `convolution.py` learns nine *continuous* weights, and enumeration does
not address a continuous part at all -- `SearchResult.continuous` reports it and
declines.  The comparable discrete question is the same nine taps with each
weight drawn from a small pool.  This measures the enumeration rate on a prefix
of that space and projects the exhaustive cost, which is the same accounting
`research/discrete-perception` uses for its flat control.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import (FMAG, IDX, MAG8, SOBEL_X, STENCIL, Builder, dump, examples, record_type,
                    report, signals)
from tcn.operators import Registry
from tcn.search import enumerate_fit, space_size
from tcn.types import Value

POOL = (-2., -1., 0., 1., 2.)


def discrete_kernel_scaffold(registry, resolution, pool=POOL, channel=1):
    """Nine taps at fixed stencil addresses, each weight a discrete choice."""
    consts = tuple((f"off_{dy}{dx}", Value.of(IDX, 3 * (dy * resolution + dx) + channel))
                   for dy, dx in STENCIL)
    consts += tuple((f"wv{i}", Value.of(FMAG, w)) for i, w in enumerate(pool))
    b = Builder(registry, (("rec", record_type(resolution)),), consts)
    b.add("pos", "project", ["rec"], params={"index": 0})
    b.add("obs", "project", ["rec"], params={"index": 1})
    terms = []
    for dy, dx in STENCIL:
        t = f"{dy}{dx}"
        b.add(f"addr_{t}", "add", ["pos", f"off_{dy}{dx}"])
        b.add(f"byte_{t}", "index", ["obs", f"addr_{t}"])
        b.add(f"mag_{t}", "interpret", [f"byte_{t}"], out=MAG8)
        b.add(f"x_{t}", "decode", [f"mag_{t}"], out=FMAG)
        b.choice(f"w_{t}", [("identity", (f"wv{i}",), None, None) for i in range(len(pool))])
        terms.append(b.add(f"term_{t}", "mul", [f"x_{t}", f"w_{t}"]))
    b.add("packed", "tuple", terms)
    b.add("conv", "sum", ["packed"])
    return b.program((("y", "conv"),))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=8)
    ap.add_argument("--train", type=int, default=3)
    ap.add_argument("--budget", type=int, default=40000)
    ap.add_argument("--tag", default="enum_projection")
    args = ap.parse_args()

    r = Registry()
    sigs = signals()
    rows = examples(tuple(range(args.train)), args.resolution, SOBEL_X, "train")
    prog = discrete_kernel_scaffold(r, args.resolution)
    n = space_size(prog)
    t0 = time.perf_counter()
    res = enumerate_fit(prog, rows, sigs, registry=r, tolerance=1e-6, max_programs=args.budget)
    wall = time.perf_counter() - t0
    rate = res.evaluated / max(1e-9, wall)
    out = {"arguments": vars(args), "rows": len(rows), "space_size": n,
           "evaluated": res.evaluated, "exhausted": res.exhausted, "seconds": wall,
           "programs_per_second": rate, "projected_exhaustive_seconds": n / max(1e-9, rate),
           "solved_in_prefix": res.solved, "conforming_in_prefix": res.conforming,
           "continuous": list(res.continuous)}
    report("discrete 3x3 kernel space (5 weights, 9 taps)", f"{n:,}")
    report("evaluated / exhausted / seconds / rate",
           f"{res.evaluated} / {res.exhausted} / {wall:.1f} / {rate:.0f} programs/s")
    report("projected exhaustive wall clock (s)", f"{out['projected_exhaustive_seconds']:.4g}")
    report("conforming found in the prefix", res.conforming)
    dump(args.tag, out)


if __name__ == "__main__":
    main()
