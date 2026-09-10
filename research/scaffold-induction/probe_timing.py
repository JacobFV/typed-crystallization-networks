"""Schedulability probe ONLY.  Shapes, space sizes, edit counts and timings.

No conformance count, accuracy or member from this file is read into any claim;
it exists to decide what the track can afford.  Disclosed in PREREGISTRATION.md
section 0.
"""
from __future__ import annotations
import sys
import time
from collections import Counter

import kit  # noqa: F401  (puts ROOT on sys.path)
import domains
import edits
from tcn.search import enumerate_prefix, space_size

CAP = 400_000


def probe(name, sample=24):
    t = time.perf_counter()
    d = domains.BUILDERS[name]()
    p, r = d["program"], d["registry"]
    allep = d["train"] + d["heldout"]
    print(f"== {name}  build {time.perf_counter() - t:.2f}s  nodes {len(p.nodes)}  "
          f"space {space_size(p)}  train {len(d['train'])} heldout {len(d['heldout'])}")
    t = time.perf_counter()
    res = enumerate_prefix(p, allep, d["signals"], r, tolerance=1e-6,
                           max_programs=1 << 24, stop_at_first=True)
    print(f"   base all-episode sweep {time.perf_counter() - t:.2f}s "
          f"conforming {res.conforming} cert {res.certificate}")
    t = time.perf_counter()
    es = edits.enumerate_edits(p, r)
    print(f"   edits {len(es)} in {time.perf_counter() - t:.2f}s  "
          f"{dict(Counter(e.family for e, _ in es))}")
    sizes = sorted(space_size(q) for _, q in es)
    over = sum(1 for s in sizes if s > CAP)
    print(f"   spaces: median {sizes[len(sizes) // 2]} max {sizes[-1]}  "
          f"over {CAP}: {over}/{len(sizes)}")
    step = max(1, len(es) // sample)
    tot, n, worst = 0.0, 0, 0.0
    for e, q in es[::step]:
        if space_size(q) > CAP:
            continue
        t = time.perf_counter()
        enumerate_prefix(q, allep, d["signals"], r, tolerance=1e-6,
                         max_programs=1 << 24, stop_at_first=True)
        dt = time.perf_counter() - t
        tot += dt
        worst = max(worst, dt)
        n += 1
    if n:
        print(f"   sampled {n} edits: mean {tot / n:.2f}s worst {worst:.2f}s  "
              f"=> whole space ~{(tot / n) * (len(es) - over) / 60:.1f} min")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["bool", "arith", "rel", "lang"]):
        try:
            probe(nm)
        except Exception:                       # noqa: BLE001
            import traceback
            traceback.print_exc()
