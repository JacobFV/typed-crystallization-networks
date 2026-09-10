"""PREREGISTRATION section 1 -- what the `min(., 3069)` clamp actually guarantees.

Two independent checks, and neither is an argument:

* **static** -- the reformulated clamp's identity `min(p + d, L) == p + min(d, L - p)`
  is verified **exhaustively** over the full reachable product of positions,
  offsets and multipliers, and over the whole rectangle `[0, L] x [0, L]` as
  well, so the reformulation of PREREGISTRATION P1.3 is a theorem about this
  scaffold rather than a plausible rewrite.  Section 56's R5a is what happens
  when a plausible rewrite is not checked.
* **dynamic** -- every node of the frozen `visual` program whose declared type is
  `IDX` (the address carrier a bound would narrow) is executed on **24 held-out
  screenshots** and its observed range recorded, together with the derived
  `a + 1` and `a + 2` that `same` computes from every address handed to it.

Nothing is declared, changed or compiled here.  This file only measures the
arm-A0 program exactly as `main` freezes it.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "research" / "compiled-runtime")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

SEEDS = tuple(range(200, 224))          # the 24 held-out screenshots of section 59's gate


# ---------------------------------------------------------------------------
# static: the reformulated clamp
# ---------------------------------------------------------------------------
def clamp_identity(width=32, height=32, span=32, offsets=(6, 99, 3, 96, 9)):
    """`min(p + d, L) == p + min(d, L - p)` -- exhaustive, two domains."""
    L = 3 * width * height - 3                       # 3069, the scaffold's `last`
    interior = [3 * (y * width + x) for y in range(1, height) for x in range(1, width)]
    every = [3 * (y * width + x) for y in range(height) for x in range(width)]
    ds = sorted({s * k for s in offsets for k in range(1, span)})

    def sweep(ps, ds):
        n = bad = 0
        pre_hi = post_hi = 0
        for p in ps:
            for d in ds:
                n += 1
                pre_hi = max(pre_hi, p + d)          # what arm A0 computes before the clamp
                a = min(p + d, L)
                b = p + min(d, L - p)
                post_hi = max(post_hi, b)
                if a != b:
                    bad += 1
        return {"pairs": n, "mismatches": bad, "max_preclamp_value": pre_hi,
                "max_reformulated_value": post_hi}

    reachable_interior = sweep(interior, ds)
    reachable_all = sweep(every, ds)
    # the full rectangle, independent of which offsets the pool happens to hold
    rect = sweep(range(0, L + 1), range(0, L + 1))
    return {"last": L, "n_offsets": len(offsets), "span": span,
            "distinct_d": len(ds), "max_d": max(ds),
            "reachable_interior_961": reachable_interior,
            "reachable_all_1024": reachable_all,
            "full_rectangle_0_to_L": rect,
            "identity_holds_everywhere": (reachable_interior["mismatches"] == 0
                                          and reachable_all["mismatches"] == 0
                                          and rect["mismatches"] == 0)}


# ---------------------------------------------------------------------------
# dynamic: every IDX-typed node of the frozen program, on 24 screenshots
# ---------------------------------------------------------------------------
def _idx_nodes(program, idx_type):
    return tuple(n.name for n in program.nodes if n.output == idx_type)


def observe(seeds=SEEDS, variant="widgets"):
    """Run the frozen modules directly and record every IDX-typed node's range.

    `Program.execute` returns the whole value environment, so this is the actual
    frozen program's actual values -- not a re-implementation.  Module calls are
    opaque to it, so the addresses handed to `same` are recorded at the call
    sites and its internal `a + 1` / `a + 2` derived from them, which is exactly
    what `same_scaffold` computes.
    """
    import fixtures                                   # noqa: F401  sets sys.path for the track
    fixtures._use("visual-ladder")
    from common import FLAT, IDX, Registry, bytes_type, episode, load, record_type
    from tcn.types import Value
    import rung3_widgets as R

    if variant == "widgets":
        found = load("rung3")
        positions_of = R.interior_positions
        corner_builder = R.corner_scaffold
        rect_builder = R.rect_scaffold
    else:
        import rung3_root as RR
        found = load("rung3_root")
        positions_of = RR.all_positions
        corner_builder = RR.corner_scaffold_masked
        rect_builder = RR.rect_scaffold_clamped

    registry = Registry()
    probe = episode(0, "train", **FLAT)
    W, H = probe["width"], probe["height"]
    offsets = R.offset_pool(W)
    p0 = R.same_scaffold(registry, W, H)
    s0sel = found["s0"]["chosen"] if variant == "widgets" else found["s0_selections"]
    m0 = registry.register_module(p0.harden(s0sel))
    p1 = corner_builder(registry, W, H, m0, offsets).harden(found["s1"]["chosen"])
    m1 = registry.register_module(p1)
    p2 = rect_builder(registry, W, H, m0, offsets).harden(found["s2"]["chosen"])
    m2 = registry.register_module(p2)

    corner_idx = _idx_nodes(p1, IDX)
    rect_idx = _idx_nodes(p2, IDX)
    # the node names whose value is an address passed to the `same` module
    same_arg_nodes = sorted({c.sources[0] for n in p1.nodes for c in n.candidates
                             if c.operator.name == m0} |
                            {c.sources[1] for n in p1.nodes for c in n.candidates
                             if c.operator.name == m0})
    same_arg_nodes_rect = sorted({s for n in p2.nodes for c in n.candidates
                                  if c.operator.name == m0 for s in c.sources[:2]})

    rng = {}

    def note(scope, name, v):
        k = "%s.%s" % (scope, name)
        lo, hi = rng.get(k, (v, v))
        rng[k] = (min(lo, v), max(hi, v))

    addresses = set()
    n_corner = n_rect = 0
    started = time.perf_counter()
    for seed in seeds:
        ep = episode(seed, "test", **FLAT)
        BT = bytes_type(ep["width"], ep["height"])
        REC = record_type(ep["width"], ep["height"])
        raw = Value.of(BT, ep["pixels"]).raw
        for pos in positions_of(ep):
            out, _, env = p1.execute({"rec": Value(REC, (pos, raw))}, registry=registry)
            n_corner += 1
            for name in corner_idx:
                note("corner", name, env[name].decoded)
            for name in same_arg_nodes:
                addresses.add(env[name].decoded)
            if out["y"].decoded:
                out2, _, env2 = p2.execute({"rec": Value(REC, (pos, raw))}, registry=registry)
                n_rect += 1
                for name in rect_idx:
                    note("rect", name, env2[name].decoded)
                for name in same_arg_nodes_rect:
                    addresses.add(env2[name].decoded)
    wall = time.perf_counter() - started

    per_node = {k: {"min": v[0], "max": v[1]} for k, v in sorted(rng.items())}
    lo = min(v["min"] for v in per_node.values())
    hi = max(v["max"] for v in per_node.values())
    a_lo, a_hi = min(addresses), max(addresses)
    return {"variant": variant, "seeds": list(seeds), "screenshots": len(seeds),
            "corner_calls": n_corner, "rect_calls": n_rect, "seconds": wall,
            "idx_nodes_corner": list(corner_idx), "idx_nodes_rect": list(rect_idx),
            "n_idx_nodes": len(corner_idx) + len(rect_idx),
            "same_address_sources": same_arg_nodes + same_arg_nodes_rect,
            "address_min": a_lo, "address_max": a_hi,
            "derived_a_plus_2_max": a_hi + 2,
            "observed_min": lo, "observed_max": hi,
            "per_node": per_node}


def main(argv):
    variants = argv[1:] or ["widgets", "root"]
    report = {"clamp_identity": clamp_identity(),
              "raster_bytes": 3 * 32 * 32,
              "candidate_bounds": {"section_59_proposal": [0, 3069],
                                   "this_track": [0, 3071]}}
    for v in variants:
        report[v] = observe(variant=v)
    (OUT / "soundness.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    ci = report["clamp_identity"]
    print("clamp identity holds everywhere:", ci["identity_holds_everywhere"])
    print("  max pre-clamp value, reachable interior:",
          ci["reachable_interior_961"]["max_preclamp_value"])
    print("  max reformulated value, reachable interior:",
          ci["reachable_interior_961"]["max_reformulated_value"])
    for v in variants:
        r = report[v]
        print("%-8s IDX nodes %d  observed [%d, %d]  addresses [%d, %d]  a+2 max %d" %
              (v, r["n_idx_nodes"], r["observed_min"], r["observed_max"],
               r["address_min"], r["address_max"], r["derived_a_plus_2_max"]))


if __name__ == "__main__":
    main(sys.argv)
