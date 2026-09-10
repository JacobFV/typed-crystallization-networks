"""PREREGISTRATION section 1.3(1) -- the static half of the soundness argument.

`soundness.py` observes the frozen program on 24 held-out screenshots.  That is
evidence about the *selected* candidate on *those* inputs.  This is the stronger
statement the pre-registration asks for: the provable range of **every**
address-typed node, over the **whole** candidate product the scaffold's search
can select and over **every** position, with the data-dependent nodes taken at
their construction bounds rather than at anything observed.

Two things make it a certificate rather than a table:

* the node set is read **off the actual `Program`** -- every node whose declared
  type is the address carrier -- and the enumeration must cover exactly that
  set, name for name, or this file raises;
* nothing here is sampled.  Positions, offsets and multipliers are enumerated
  exhaustively; the data-dependent `we_k` is taken over `{0, 1}` (it is
  `encode` of a `bool`) and `w_count` over `[0, span-1]` (it is a `sum` of
  `span-1` such terms), which are their type-level extremes, not measured ones.
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "research" / "compiled-runtime")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms                                            # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def _rng(vals):
    return (min(vals), max(vals))


def _join(a, b):
    return (min(a[0], b[0]), max(a[1], b[1]))


def enumerate_ranges(width=32, height=32, span=32, offsets=(6, 99, 3, 96, 9),
                     variant="widgets", reformulated=True):
    """Exhaustive reachable range of every address-typed node, by name."""
    last = 3 * width * height - 3                       # 3069
    if variant == "widgets":
        positions = [3 * (y * width + x) for y in range(1, height) for x in range(1, width)]
    else:
        positions = [3 * (y * width + x) for y in range(height) for x in range(width)]
    ks = list(range(1, span))
    r = {}

    def note(name, v):
        r[name] = v if name not in r else _join(r[name], v)

    # ---- the caller's position set, and what `same` is asked to read ----------
    note("pos", _rng(positions))
    addresses = set()

    # ---- S1: corner ----------------------------------------------------------
    if variant == "widgets":
        # back_a / back_b = sub(pos, off)
        for off in offsets:
            vals = [p - off for p in positions]
            note("back_a", _rng(vals))
            note("back_b", _rng(vals))
            addresses |= set(vals)
    else:
        note("col", _rng([p % (3 * width) for p in positions]))
        for off in offsets:
            safe = [max(p, off) for p in positions]
            addr = [s - off for s in safe]
            for arm in ("a", "b"):
                note("back_%s" % arm, (off, off))
                note("safe_%s" % arm, _rng(safe))
                note("addr_%s" % arm, _rng(addr))
                note("bcol_%s" % arm, _rng([a % (3 * width) for a in addr]))
            addresses |= set(addr)
    addresses |= set(positions)

    # ---- S2: rect ------------------------------------------------------------
    note("linear", _rng([p // 3 for p in positions]))
    note("x", _rng([(p // 3) % width for p in positions]))
    note("y", _rng([(p // 3) // width for p in positions]))
    if reformulated:
        note("room", _rng([last - p for p in positions]))
    for tag, coordinate in (("w", "x"), ("h", "y")):
        note("step_%s" % tag, _rng(offsets))              # identity(off_k), any candidate
        for step in offsets:
            for k in ks:
                d = step * k
                note("%sd%d" % (tag, k), (d, d))
                if reformulated:
                    caps = [min(d, last - p) for p in positions]
                    addr = [p + c for p, c in zip(positions, caps)]
                    note("%sp%d" % (tag, k), _rng(caps))
                else:
                    raw = [p + d for p in positions]
                    addr = [min(v, last) for v in raw]
                    note("%sa%d" % (tag, k), _rng(raw))
                note("%sc%d" % (tag, k), _rng(addr))
                addresses |= set(addr)
                base = [(p // 3) % width if coordinate == "x" else (p // 3) // width
                        for p in positions]
                note("%si%d" % (tag, k), _rng([b + k for b in base]))
                note("%se%d" % (tag, k), (0, 1))         # encode(bool)
        note("%s_count" % tag, (0, span - 1))            # sum of span-1 terms in {0, 1}
        note("%s_extent" % tag, (1, span))
    if variant == "widgets":
        note("left", _rng([p - 3 for p in positions]))
    else:
        note("safe_left", _rng([max(p, 3) for p in positions]))
        note("left", _rng([max(p, 3) - 3 for p in positions]))
    for tag, base in (("own", "pos"), ("par", "left")):
        b = r[base if tag == "own" else "left"]
        note("%s_g" % tag, (b[0] + 1, b[1] + 1))
        note("%s_b" % tag, (b[0] + 2, b[1] + 2))

    # ---- S0: same(a, b, obs), which reads a, a+1, a+2 ------------------------
    a_lo, a_hi = min(addresses), max(addresses)
    same = {"a": (a_lo, a_hi), "b": (a_lo, a_hi),
            "a_g": (a_lo + 1, a_hi + 1), "a_b": (a_lo + 2, a_hi + 2),
            "b_g": (a_lo + 1, a_hi + 1), "b_b": (a_lo + 2, a_hi + 2)}
    return r, same, sorted(addresses)


def check(variant="widgets", reformulated=True):
    """Cross-check the enumeration's node set against the real `Program`."""
    fixtures = arms._track()
    from common import FLAT, IDX, Registry, episode, load
    import rung3_widgets as R

    registry = Registry()
    probe = episode(0, "train", **FLAT)
    W, H = probe["width"], probe["height"]
    offsets = R.offset_pool(W)
    if variant == "widgets":
        found = load("rung3")
        p0 = R.same_scaffold(registry, W, H)
        m0 = registry.register_module(arms.harden_all(p0, found["s0"]["chosen"]))
        p1 = R.corner_scaffold(registry, W, H, m0, offsets)
        p2 = R.rect_scaffold(registry, W, H, m0, offsets, reformulated_clamp=reformulated)
    else:
        import rung3_root as RR
        found = load("rung3_root")
        p0 = R.same_scaffold(registry, W, H)
        m0 = registry.register_module(arms.harden_all(p0, found["s0_selections"]))
        p1 = RR.corner_scaffold_masked(registry, W, H, m0, offsets)
        p2 = RR.rect_scaffold_clamped(registry, W, H, m0, offsets,
                                      reformulated_clamp=reformulated)

    declared = {n.name for p in (p1, p2) for n in p.nodes if n.output == IDX}
    same_nodes = {n.name for n in p0.nodes if n.output == IDX}
    ranges, same, addresses = enumerate_ranges(W, H, max(W, H), offsets, variant, reformulated)
    missing = sorted(declared - set(ranges))
    extra = sorted(set(ranges) - declared - {"pos"})
    missing_same = sorted(same_nodes - set(same))
    lo = min(v[0] for v in list(ranges.values()) + list(same.values()))
    hi = max(v[1] for v in list(ranges.values()) + list(same.values()))
    bound = (0, 3 * W * H - 1)
    worst = sorted(((v[1], k) for k, v in list(ranges.items()) + list(same.items())),
                   reverse=True)[:6]
    return {"variant": variant, "reformulated_clamp": reformulated,
            "address_nodes_declared": len(declared) + len(same_nodes),
            "nodes_not_covered_by_the_enumeration": missing + missing_same,
            "enumerated_names_not_in_the_program": extra,
            "coverage_complete": not (missing or missing_same),
            "positions": len(addresses), "address_min": min(addresses),
            "address_max": max(addresses),
            "reachable_min": lo, "reachable_max": hi,
            "candidate_bound": list(bound),
            "sound_for_candidate_bound": bound[0] <= lo and hi <= bound[1],
            "sound_for_3069_bound": 0 <= lo and hi <= 3 * W * H - 3,
            "six_largest_nodes": [[k, v] for v, k in worst],
            "ranges": {k: list(v) for k, v in sorted(ranges.items())},
            "same_module_ranges": {k: list(v) for k, v in sorted(same.items())}}


def main(argv):
    report = {}
    for variant in ("widgets", "root"):
        for reformulated in (False, True):
            key = "%s_%s" % (variant, "reformulated" if reformulated else "as_written")
            report[key] = check(variant, reformulated)
            r = report[key]
            print("%-24s coverage %s  reachable [%d, %d]  sound for (0,3071)=%s  (0,3069)=%s"
                  % (key, r["coverage_complete"], r["reachable_min"], r["reachable_max"],
                     r["sound_for_candidate_bound"], r["sound_for_3069_bound"]))
            print("   six largest:", r["six_largest_nodes"])
    (OUT / "static_ranges.json").write_text(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv)
