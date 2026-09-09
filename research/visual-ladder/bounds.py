"""Every recoverability bound this track relies on, computed before any search.

The method is `research/object-identity`'s and `research/gui-hierarchy`'s: a
lookup table keyed by exactly the context in question is an upper bound on
*every* function of that context, so a ceiling at the majority baseline is a
certificate that no program over that context can beat a constant.

Two families of bound are computed here, and they answer different questions:

* an **informational** bound (`lookup_bound`) -- can any function of this context
  produce the target;
* an **exactness check** on a specific, stated rule -- does *this* rule reproduce
  the target at every instance, which is what a program has to do.

The second kind is what rung three needs, because its candidate rules are not
statistical: "the pixel left of a widget's top-left corner belongs to its parent"
is either true everywhere or it is not a rule.
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import (FLAT, TEXT, colour_at, dump, episode, ink_grid, is_ink, lookup_bound,
                    report, window_bits, window_bytes)
from generators.gui.render import KINDS

ALPHABET_SIZE = 36


# --------------------------------------------------------------------------
# rung two: glyphs
# --------------------------------------------------------------------------
def glyph_records(seeds, split, window, **configuration):
    """One record per drawn glyph: the ink window at its own anchor, and its code.

    The **anchor** is the glyph's bounding-box top-left.  It is taken from the
    probe here because this is a bound, not a program: the question "is the code
    determined by the window" is separate from "can the anchor be found", which
    `anchor_records` bounds directly below.
    """
    ww, hh = window
    rows = []
    for seed in seeds:
        ep = episode(seed, split, **configuration)
        grid = ink_grid(ep)
        for g in ep["probes"]["glyphs"]:
            x, y = g["rect"][0], g["rect"][1]
            rows.append({"seed": seed, "code": g["code"], "rect": tuple(g["rect"]),
                         "ink": window_bits(grid, x, y, ww, hh, ep["width"], ep["height"]),
                         "raw": window_bytes(ep, x, y, ww, hh)})
    return rows


def anchor_records(seeds, split, window, **configuration):
    """One record per screen position: the ink window, and whether it is an anchor."""
    ww, hh = window
    rows = []
    for seed in seeds:
        ep = episode(seed, split, **configuration)
        grid = ink_grid(ep)
        anchors = {(g["rect"][0], g["rect"][1]) for g in ep["probes"]["glyphs"]}
        for y in range(ep["height"]):
            for x in range(ep["width"]):
                if not grid[y][x]:
                    continue          # an anchor is an ink pixel by construction
                rows.append((window_bits(grid, x, y, ww, hh, ep["width"], ep["height"]),
                             (x, y) in anchors))
    return rows


def linear_family_bound(rows):
    """Is the code an exact linear functional of the ink window, over R?

    A bound on the *continuous* family the `interpret` commitment opens: if no
    weight vector reproduces the code exactly on the training patterns, no
    program of the form `sum_k w_k * decode(interpret(ink_k))` can, whatever the
    optimiser does.  Solved by least squares outside the substrate; the residual
    is the bound and the recovered weights are not used anywhere.
    """
    import numpy as np
    patterns = {}
    for r in rows:
        patterns.setdefault(r["ink"], set()).add(r["code"])
    collisions = {k: sorted(v) for k, v in patterns.items() if len(v) > 1}
    keys = sorted(patterns)
    a = np.array([[1.] + list(k) for k in keys], dtype=float)
    b = np.array([sorted(patterns[k])[0] for k in keys], dtype=float)
    w, *_ = np.linalg.lstsq(a, b, rcond=None)
    residual = float(np.max(np.abs(a @ w - b))) if len(keys) else 0.
    return {"distinct_patterns": len(keys), "features": a.shape[1],
            "matrix_rank": int(np.linalg.matrix_rank(a)),
            "colliding_patterns": len(collisions),
            "collision_examples": [v for v in collisions.values()][:5],
            "max_abs_residual": residual,
            "exactly_linear": residual < .5,
            "note": "a residual below 0.5 rounds to the exact integer code"}


def bit_family_bound(rows, bits=6):
    """Best 1-address and best 2-address Boolean predictor of each code bit.

    The small discrete family the algebra can enumerate: one ink bit, or a
    two-input truth table over two ink bits.  This is a bound on that family and
    not on the algebra, which is exactly the distinction `research/FINDINGS.md`
    section 14 draws between an informational wall and an exhausted family.
    """
    n = len(rows[0]["ink"])
    out = []
    for bit in range(bits):
        labels = [(r["code"] >> bit) & 1 for r in rows]
        majority = max(sum(labels), len(labels) - sum(labels)) / len(labels)
        best1 = max(max(sum(r["ink"][a] == l for r, l in zip(rows, labels)),
                        sum(r["ink"][a] != l for r, l in zip(rows, labels))) / len(labels)
                    for a in range(n))
        best2 = 0.
        for a in range(n):
            for c in range(a + 1, n):
                counts = collections.Counter((r["ink"][a], r["ink"][c], l)
                                             for r, l in zip(rows, labels))
                hit = sum(max(counts[(p, q, 0)], counts[(p, q, 1)])
                          for p in (0, 1) for q in (0, 1))
                best2 = max(best2, hit / len(labels))
        out.append({"bit": bit, "majority": majority, "best_one_address": best1,
                    "best_two_address": best2, "addresses": n})
    return out


# --------------------------------------------------------------------------
# rung three: widgets
# --------------------------------------------------------------------------
def widget_records(seeds, split, **configuration):
    rows = []
    for seed in seeds:
        ep = episode(seed, split, **configuration)
        for w in ep["probes"]["hierarchy"]:
            x, y, ww, hh = w["rect"]
            rows.append({"seed": seed, "id": w["id"], "parent": w["parent"], "kind": w["kind"],
                         "rect": (x, y, ww, hh), "colour": colour_at(ep, x, y)})
    return rows


def corner_rule(ep):
    """`p` is its widget's top-left corner iff owner differs left AND above.

    Stated as an exact rule and checked at every position, because a rule that is
    right 99% of the time is not a program.  The rule is expressible as two calls
    to the frozen rung-one edge module at computed offsets.
    """
    owner, w, h = ep["probes"]["owner"], ep["width"], ep["height"]
    truth = {(x, y) for x, y, *_ in
             [(d["rect"][0], d["rect"][1]) for d in ep["probes"]["hierarchy"]]}
    tp = fp = fn = tn = 0
    for y in range(h):
        for x in range(w):
            i = y * w + x
            left = x == 0 or owner[i] != owner[i - 1]
            above = y == 0 or owner[i] != owner[i - w]
            predicted = left and above
            actual = (x, y) in truth
            tp += predicted and actual
            fp += predicted and not actual
            fn += (not predicted) and actual
            tn += (not predicted) and not actual
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "true_negative": tn, "positions": w * h, "corners": len(truth)}


def colour_corner_rule(ep):
    """The same rule with *colour* inequality instead of the owner probe.

    This is what a program actually computes -- the rung-one module compares
    pixels, not owners -- so the gap between the two is the price of the
    substitution and it is measured rather than assumed.
    """
    w, h = ep["width"], ep["height"]
    truth = {(d["rect"][0], d["rect"][1]) for d in ep["probes"]["hierarchy"]}
    tp = fp = fn = 0
    for y in range(h):
        for x in range(w):
            left = x == 0 or colour_at(ep, x - 1, y) != colour_at(ep, x, y)
            above = y == 0 or colour_at(ep, x, y - 1) != colour_at(ep, x, y)
            predicted = left and above
            actual = (x, y) in truth
            tp += predicted and actual
            fp += predicted and not actual
            fn += (not predicted) and actual
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "positions": w * h, "corners": len(truth)}


def extent_rule(ep):
    """At a corner, the same-colour run right/down is the widget's width/height.

    `layout` insets every child by `margin`, so a widget's own top row and left
    column are never covered by a child; the run therefore stops exactly at the
    rectangle's far edge, provided the neighbouring widget has a different
    colour.  Checked, not assumed.
    """
    w, h = ep["width"], ep["height"]
    hits = misses = 0
    detail = []
    for d in ep["probes"]["hierarchy"]:
        x, y, ww, hh = d["rect"]
        base = colour_at(ep, x, y)
        run_x = 1
        while x + run_x < w and colour_at(ep, x + run_x, y) == base:
            run_x += 1
        run_y = 1
        while y + run_y < h and colour_at(ep, x, y + run_y) == base:
            run_y += 1
        ok = (run_x == ww) and (run_y == hh)
        hits += ok
        misses += not ok
        if not ok:
            detail.append({"id": d["id"], "rect": d["rect"], "run": (run_x, run_y)})
    return {"widgets": hits + misses, "exact": hits, "wrong": misses, "examples": detail[:5]}


def parent_pixel_rule(ep):
    """The pixel immediately left of a widget's top-left corner belongs to its parent."""
    owner, w = ep["probes"]["owner"], ep["width"]
    hits = misses = 0
    detail = []
    for d in ep["probes"]["hierarchy"]:
        x, y = d["rect"][0], d["rect"][1]
        if d["parent"] == 255:
            continue
        got = owner[y * w + (x - 1)] if x > 0 else None
        hits += got == d["parent"]
        misses += got != d["parent"]
        if got != d["parent"]:
            detail.append({"id": d["id"], "expected": d["parent"], "got": got, "rect": d["rect"]})
    return {"non_root": hits + misses, "exact": hits, "wrong": misses, "examples": detail[:5]}


def smallest_container_rule(ep):
    """`research/gui-hierarchy` R4, re-checked on this track's screen."""
    rects = {d["id"]: d["rect"] for d in ep["probes"]["hierarchy"]}
    hits = misses = ties = 0
    for d in ep["probes"]["hierarchy"]:
        if d["parent"] == 255:
            continue
        x, y, w, h = d["rect"]
        best, area = None, None
        for i, (a, b, c, e) in rects.items():
            if i == d["id"]:
                continue
            if a <= x and b <= y and a + c >= x + w and b + e >= y + h:
                if area is None or c * e < area:
                    best, area, ties = i, c * e, ties
                elif c * e == area:
                    ties += 1
        hits += best == d["parent"]
        misses += best != d["parent"]
    return {"non_root": hits + misses, "exact": hits, "wrong": misses, "ties": ties}


def colour_injectivity(ep):
    seen = collections.Counter()
    for d in ep["probes"]["hierarchy"]:
        seen[colour_at(ep, d["rect"][0], d["rect"][1])] += 1
    return {"widgets": len(ep["probes"]["hierarchy"]), "distinct_colours": len(seen),
            "injective": all(v == 1 for v in seen.values())}


def rule_totals(rows):
    total = collections.Counter()
    for r in rows:
        for k, v in r.items():
            if isinstance(v, int):
                total[k] += v
    return dict(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=12)
    ap.add_argument("--held", type=int, default=12)
    ap.add_argument("--tag", default="bounds")
    args = ap.parse_args()
    result = {"arguments": vars(args), "flat_screen": FLAT, "text_screen": TEXT}

    # ---- rung two -------------------------------------------------------
    print("\n=== rung two: glyphs ===")
    train_seeds, held_seeds = range(args.train), range(500, 500 + args.held)
    result["glyph_windows"] = []
    cache = {}
    for window in ((6, 5), (5, 5), (4, 5), (8, 6), (3, 5)):
        tr = cache.setdefault(("tr", window), glyph_records(train_seeds, "train", window, **TEXT))
        ev = cache.setdefault(("ev", window), glyph_records(held_seeds, "test", window, **TEXT))
        for reduction in ("ink", "raw"):
            bound = lookup_bound([(r[reduction], r["code"]) for r in tr],
                                 [(r[reduction], r["code"]) for r in ev])
            bound |= {"window": list(window), "reduction": reduction}
            result["glyph_windows"].append(bound)
            report(f"glyph code | {window} {reduction:3s} majority/oracle/transfer",
                   f"{bound['majority_baseline']:.4f} / {bound['oracle_accuracy']:.4f} / "
                   f"{bound['transfer_accuracy']:.4f} (unseen {bound['unseen_key_fraction']:.3f})")

    best = (6, 5)
    tr, ev = cache[("tr", best)], cache[("ev", best)]
    result["glyph_counts"] = {"train_glyphs": len(tr), "held_glyphs": len(ev),
                              "distinct_codes_train": len({r["code"] for r in tr}),
                              "distinct_codes_held": len({r["code"] for r in ev}),
                              "window": list(best)}
    result["linear_family"] = linear_family_bound(tr)
    report("glyph code | linear family: rank / patterns / residual",
           f"{result['linear_family']['matrix_rank']} / "
           f"{result['linear_family']['distinct_patterns']} / "
           f"{result['linear_family']['max_abs_residual']:.4g}")
    result["bit_family"] = bit_family_bound(tr)
    for row in result["bit_family"]:
        report(f"glyph code bit {row['bit']} | majority / best 1-addr / best 2-addr",
               f"{row['majority']:.4f} / {row['best_one_address']:.4f} / "
               f"{row['best_two_address']:.4f}")

    result["glyph_anchor"] = []
    for window in ((3, 3), (5, 5)):
        tr_a = anchor_records(train_seeds, "train", window, **TEXT)
        ev_a = anchor_records(held_seeds, "test", window, **TEXT)
        bound = lookup_bound(tr_a, ev_a) | {"window": list(window)}
        result["glyph_anchor"].append(bound)
        report(f"glyph anchor | {window} ink majority/oracle/transfer",
               f"{bound['majority_baseline']:.4f} / {bound['oracle_accuracy']:.4f} / "
               f"{bound['transfer_accuracy']:.4f}")

    # ---- rung three -----------------------------------------------------
    print("\n=== rung three: widgets ===")
    wtr = widget_records(train_seeds, "train", **FLAT)
    wev = widget_records(held_seeds, "test", **FLAT)
    result["widget_kind"] = []
    for name, key in (("fill colour", lambda r: r["colour"]),
                      ("size (w,h)", lambda r: r["rect"][2:]),
                      ("fill + size", lambda r: r["colour"] + r["rect"][2:]),
                      ("position + size", lambda r: r["rect"])):
        bound = lookup_bound([(key(r), r["kind"]) for r in wtr],
                             [(key(r), r["kind"]) for r in wev]) | {"context": name}
        result["widget_kind"].append(bound)
        report(f"widget kind | {name:16s} majority/oracle/transfer",
               f"{bound['majority_baseline']:.4f} / {bound['oracle_accuracy']:.4f} / "
               f"{bound['transfer_accuracy']:.4f}")

    result["rules"] = {}
    for label, configuration in (("flat", FLAT), ("borders", FLAT | {"borders": True}),
                                 ("labels", FLAT | {"labels": True, "label_size": 8}),
                                 ("palette 8", FLAT | {"palette": 8}),
                                 ("text screen", TEXT),
                                 ("text screen, labels off", TEXT | {"labels": False})):
        eps = [episode(s, "train", **configuration) for s in train_seeds]
        block = {"configuration": configuration,
                 "achieved_widgets": sum(len(e["probes"]["hierarchy"]) for e in eps) / len(eps),
                 "corner_owner": rule_totals([corner_rule(e) for e in eps]),
                 "corner_colour": rule_totals([colour_corner_rule(e) for e in eps]),
                 "extent": rule_totals([extent_rule(e) for e in eps]),
                 "parent_pixel": rule_totals([parent_pixel_rule(e) for e in eps]),
                 "smallest_container": rule_totals([smallest_container_rule(e) for e in eps]),
                 "colour_injective": rule_totals([colour_injectivity(e) for e in eps])}
        result["rules"][label] = block
        report(f"[{label:22s}] corner(colour) tp/fp/fn",
               f"{block['corner_colour']['true_positive']}/"
               f"{block['corner_colour']['false_positive']}/"
               f"{block['corner_colour']['false_negative']}")
        report(f"[{label:22s}] extent exact / parent-pixel exact / container exact",
               f"{block['extent']['exact']}/{block['extent']['widgets']}  "
               f"{block['parent_pixel']['exact']}/{block['parent_pixel']['non_root']}  "
               f"{block['smallest_container']['exact']}/{block['smallest_container']['non_root']}")

    dump(args.tag, result)


if __name__ == "__main__":
    main()
