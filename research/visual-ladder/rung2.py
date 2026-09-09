"""Rung two, staged on rung three's corner: the character key from a widget-anchored window.

Section 1.2 measured that a local ink window does *not* determine the glyph's own
anchor (0.982 ceiling), so a rung two built on a self-detected anchor is capped
before any program is written.  Section 1.4 measured that the *widget's* corner is
exact, and `rung3_root.py` now produces it.  So the window is anchored at the
widget's text origin, `(rect.x + 1, rect.y + 1)`, which rung three supplies.

Section 1.3's recommendation, followed here: the glyph *code* is an arbitrary
index into `ALPHABET` and both expressible families were excluded exhaustively,
so the target is not the code.  It is the **relation** the packed ink window
supports -- `same_character(a, b)` is `eq` on two keys -- which is the
object-identity survivor and is well inside the algebra.

Two searches, both exhaustible:

  T0  `ink(a, obs) -> bool`  -- 256 programs, two searched Boolean combinators
      over `eq(channel, 0)` on the three channels.  Supervised by whether the
      addressed pixel is `INK`, which is a function of the observation alone.
  T1  `same_character(rec_a, rec_b) -> bool` -- `eq(pack(w), pack(w))` on two
      8x8 ink windows taken at a **searched** anchor offset from the widget
      corner.  The pool is `offset_pool`, whose answer `3W + 3` is at index 1;
      space 5, and every candidate's accuracy is reported, not only the
      conforming ones.

`label_length=1` throughout: section 1.1 measured that a wider window reaches
into the next character of the same label and makes the key label-specific.
The known structural ceiling is 0.973 -- `i`/`j` and `f`/`l` render bit-identically
after the bounding box is trimmed -- and it is re-derived here on the pair task.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import (BYTE, IDX, TEXT, Builder, Registry, accuracy, all_conforming, bytes_type,
                    dump, episode, ink_grid, lookup_bound, record_type, report, sweep,
                    window_bits)
from generators.gui.render import ALPHABET
from tcn.search import space_size
from tcn.types import BOOL, Value, integer

WINDOW = (8, 8)
BITS = WINDOW[0] * WINDOW[1]
CKEY = integer(64, signed=False)
SCREEN = dict(TEXT) | {"label_length": 1}


# --------------------------------------------------------------------------
def widget_glyphs(ep):
    """Each labelled widget's corner, text origin and its single glyph's code.

    A glyph is assigned to the smallest widget rectangle containing it, which is
    the same containment resolution the rung-three parse uses for the parent.
    """
    widgets = ep["probes"]["hierarchy"]
    out = []
    for g in ep["probes"].get("glyphs", []):
        gx, gy = g["rect"][0], g["rect"][1]
        inside = [d for d in widgets
                  if d["rect"][0] <= gx < d["rect"][0] + d["rect"][2]
                  and d["rect"][1] <= gy < d["rect"][1] + d["rect"][3]]
        if not inside:
            continue
        owner = min(inside, key=lambda d: d["rect"][2] * d["rect"][3])
        out.append({"corner": (owner["rect"][0], owner["rect"][1]), "code": g["code"],
                    "glyph": (gx, gy)})
    # one glyph per widget at label_length 1; drop any widget that got two
    seen = {}
    for r in out:
        seen.setdefault(r["corner"], []).append(r)
    return [v[0] for v in seen.values() if len(v) == 1]


def anchored_records(seeds, split, window, **configuration):
    """The ink window at each labelled widget's text origin, and the glyph code."""
    ww, hh = window
    rows = []
    for seed in seeds:
        ep = episode(seed, split, **configuration)
        grid = ink_grid(ep)
        for r in widget_glyphs(ep):
            x, y = r["corner"][0] + 1, r["corner"][1] + 1
            rows.append({"seed": seed, "code": r["code"], "corner": r["corner"],
                         "ink": window_bits(grid, x, y, ww, hh, ep["width"], ep["height"])})
    return rows


# --------------------------------------------------------------------------
# T0: ink(a, obs)
# --------------------------------------------------------------------------
def ink_scaffold(registry, width, height):
    """`(a, obs) -> bool`: is the addressed pixel ink?  Two searched combinators."""
    BT = bytes_type(width, height)
    consts = (("one", Value.of(IDX, 1)), ("two", Value.of(IDX, 2)),
              ("zero", Value.of(BYTE, 0)))
    b = Builder(registry, (("a", IDX), ("obs", BT)), consts)
    b.add("ag", "add", ["a", "one"])
    b.add("ab", "add", ["a", "two"])
    b.add("rv", "index", ["obs", "a"])
    b.add("gv", "index", ["obs", "ag"])
    b.add("bv", "index", ["obs", "ab"])
    b.add("r0", "eq", ["rv", "zero"])
    b.add("g0", "eq", ["gv", "zero"])
    b.add("b0", "eq", ["bv", "zero"])
    b.choice("rg", [(f"truth_{t}", ("r0", "g0"), None, None) for t in range(16)])
    b.choice("ink", [(f"truth_{t}", ("rg", "b0"), None, None) for t in range(16)])
    return b.program((("y", "ink"),))


def ink_examples(seeds, split, per_image, seed=0, **configuration):
    import random
    rng = random.Random(seed)
    rows = []
    for s in seeds:
        ep = episode(s, split, **configuration)
        BT = bytes_type(ep["width"], ep["height"])
        raw = Value.of(BT, ep["pixels"]).raw
        grid = ink_grid(ep)
        w, h = ep["width"], ep["height"]
        ink = [(x, y) for y in range(h) for x in range(w) if grid[y][x]]
        other = [(x, y) for y in range(h) for x in range(w) if not grid[y][x]]
        take = rng.sample(ink, min(per_image // 2, len(ink))) + \
            rng.sample(other, min(per_image // 2, len(other)))
        for x, y in take:
            rows.append({"inputs": {"a": Value.of(IDX, 3 * (y * w + x)),
                                    "obs": Value(BT, raw)},
                         "targets": {"ink": Value.of(BOOL, grid[y][x] == 1)}})
    rng.shuffle(rows)
    return rows


def induced_ink(selections):
    def truth(t, x, y):
        return bool((t >> (2 * int(x) + int(y))) & 1)
    return tuple(truth(selections["ink"], truth(selections["rg"], a, b), c)
                 for a in (False, True) for b in (False, True) for c in (False, True))


# --------------------------------------------------------------------------
# T1: same_character(rec_a, rec_b)
# --------------------------------------------------------------------------
def same_character_scaffold(registry, width, height, module, offsets, window=WINDOW):
    """`eq` on two packed ink windows taken at a searched anchor offset."""
    ww, hh = window
    last = 3 * width * height - 3
    REC = record_type(width, height)
    consts = tuple((f"off{k}", Value.of(IDX, k)) for k in offsets)
    consts += (("last", Value.of(IDX, last)),)
    steps = sorted({3 * (dy * width + dx) for dy in range(hh) for dx in range(ww)})
    consts += tuple((f"s{v}", Value.of(IDX, v)) for v in steps if v)
    b = Builder(registry, (("ra", REC), ("rb", REC)), consts)
    b.choice("anchor", [("identity", (f"off{k}",), None, None) for k in offsets])
    for arm in ("a", "b"):
        b.add(f"pos_{arm}", "project", [f"r{arm}"], params={"index": 0})
        b.add(f"obs_{arm}", "project", [f"r{arm}"], params={"index": 1})
        b.add(f"origin_{arm}", "add", [f"pos_{arm}", "anchor"])
        bits = []
        for dy in range(hh):
            for dx in range(ww):
                v = 3 * (dy * width + dx)
                base = (f"origin_{arm}" if v == 0
                        else b.add(f"p_{arm}_{dy}_{dx}", "add", [f"origin_{arm}", f"s{v}"]))
                c = b.add(f"c_{arm}_{dy}_{dx}", "min", [base, "last"])
                bits.append(b.add(f"i_{arm}_{dy}_{dx}", module, [c, f"obs_{arm}"]))
        b.add(f"tup_{arm}", "tuple", bits)
        b.add(f"key_{arm}", "pack", [f"tup_{arm}"], out=CKEY)
    b.add("same", "eq", ["key_a", "key_b"])
    return b.program((("y", "same"),))


def pair_examples(seeds, split, pairs, seed=0, **configuration):
    """Balanced same/different character pairs, pooled **across** episodes.

    Each record carries its own observation, so a pair may span two screens.
    That is deliberate: section 1.3's claim is that the packed ink window is an
    *episode-independent* character key, and a within-screen pool would rarely
    contain two widgets labelled with the same character.
    """
    import random
    rng = random.Random(seed)
    pool = []
    for s in seeds:
        ep = episode(s, split, **configuration)
        BT = bytes_type(ep["width"], ep["height"])
        REC = record_type(ep["width"], ep["height"])
        raw = Value.of(BT, ep["pixels"]).raw
        for r in widget_glyphs(ep):
            addr = 3 * (r["corner"][1] * ep["width"] + r["corner"][0])
            pool.append({"rec": Value(REC, (addr, raw)), "code": r["code"]})
    by_code = {}
    for r in pool:
        by_code.setdefault(r["code"], []).append(r)
    repeated = [c for c, v in by_code.items() if len(v) >= 2]
    rows = []
    for i in range(pairs):
        if i % 2 == 0 and repeated:
            a, bb = rng.sample(by_code[rng.choice(repeated)], 2)
        else:
            a, bb = rng.sample(pool, 2)
        rows.append({"inputs": {"ra": a["rec"], "rb": bb["rec"]},
                     "targets": {"same": Value.of(BOOL, a["code"] == bb["code"])}})
    rng.shuffle(rows)
    return rows


def pair_signals():
    from tcn.graph import Signal
    return (Signal("same", "same", ("core",), BOOL, "bce"),)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=4)
    ap.add_argument("--held", type=int, default=4)
    ap.add_argument("--ink-per-image", type=int, default=60)
    ap.add_argument("--pairs", type=int, default=24)
    ap.add_argument("--bound-train", type=int, default=12)
    ap.add_argument("--bound-held", type=int, default=12)
    ap.add_argument("--tag", default="rung2")
    args = ap.parse_args()

    from tcn.graph import Signal
    registry = Registry()
    tolerance = 1e-6
    result = {"arguments": vars(args), "configuration": SCREEN, "window": list(WINDOW)}
    probe = episode(0, "train", **SCREEN)
    W, H = probe["width"], probe["height"]
    offsets = __import__("rung3_widgets").offset_pool(W)
    result["offset_pool"] = list(offsets)
    result["anchor_answer_index"] = offsets.index(3 * W + 3)

    # ---- the bound, re-measured at this window and this anchor ----------
    print("\n=== rung two bound: widget-anchored window -> glyph code ===")
    tr = anchored_records(range(args.bound_train), "train", WINDOW, **SCREEN)
    ev = anchored_records(range(500, 500 + args.bound_held), "test", WINDOW, **SCREEN)
    bound = lookup_bound([(r["ink"], r["code"]) for r in tr],
                         [(r["ink"], r["code"]) for r in ev])
    patterns = {}
    for r in tr + ev:
        patterns.setdefault(r["ink"], set()).add(r["code"])
    collisions = {i: sorted(v) for i, (k, v) in enumerate(patterns.items()) if len(v) > 1}
    bound |= {"train_glyph_widgets": len(tr), "held_glyph_widgets": len(ev),
              "distinct_patterns": len(patterns),
              "colliding_patterns": len(collisions),
              "colliding_code_sets": sorted({tuple(v) for v in collisions.values()}),
              "colliding_characters": sorted({tuple(ALPHABET[c] for c in v)
                                              for v in collisions.values()})}
    result["code_bound"] = bound
    report("code | majority / oracle / transfer / unseen",
           f"{bound['majority_baseline']:.4f} / {bound['oracle_accuracy']:.4f} / "
           f"{bound['transfer_accuracy']:.4f} / {bound['unseen_key_fraction']:.3f}")
    report("colliding classes", bound["colliding_characters"])

    # the same key, on the relation the algebra can actually express
    pair_bound = []
    for a in range(len(ev)):
        for bidx in range(a + 1, len(ev)):
            pair_bound.append((ev[a]["ink"] == ev[bidx]["ink"], ev[a]["code"] == ev[bidx]["code"]))
    hits = sum(p == t for p, t in pair_bound)
    result["relation_bound"] = {
        "held_pairs": len(pair_bound),
        "key_equality_accuracy": hits / max(1, len(pair_bound)),
        "always_different_baseline": sum(not t for _, t in pair_bound) / max(1, len(pair_bound)),
        "positive_fraction": sum(t for _, t in pair_bound) / max(1, len(pair_bound))}
    report("relation | key-eq accuracy / always-different baseline",
           f"{result['relation_bound']['key_equality_accuracy']:.4f} / "
           f"{result['relation_bound']['always_different_baseline']:.4f}")
    dump(args.tag, result)

    # ---- T0 -------------------------------------------------------------
    print("\n--- T0: ink(a, obs), the reduction the R2 bound identified ---")
    R3 = __import__("rung3_widgets")
    itr = ink_examples(range(args.train), "train", args.ink_per_image, seed=11, **SCREEN)
    iva = ink_examples(range(50, 50 + args.train), "validation", args.ink_per_image, seed=12,
                       **SCREEN)
    ihe = ink_examples(range(100, 100 + args.held), "test", args.ink_per_image, seed=13, **SCREEN)
    signals = (Signal("ink", "ink", ("core",), BOOL, "bce"),)
    program = ink_scaffold(registry, W, H)
    result["t0"], conforming = R3.stage("T0 ink", program, itr, iva, ihe, signals, registry,
                                        tolerance, induced=induced_ink)
    if "chosen" not in result["t0"]:
        raise SystemExit("T0: 0 conforming, exhausted -- that is the certificate.")
    ink_module = registry.register_module(program.harden(result["t0"]["chosen"]))
    dump(args.tag, result)

    # ---- T1 -------------------------------------------------------------
    print("\n--- T1: same_character, the anchor searched from rung three's corner ---")
    ptr = pair_examples(range(args.bound_train), "train", args.pairs, seed=21, **SCREEN)
    phe = pair_examples(range(500, 500 + args.bound_held), "test", args.pairs, seed=23,
                        **SCREEN)
    result["t1_positive_fraction"] = sum(e["targets"]["same"].decoded for e in ptr) / len(ptr)
    sc = same_character_scaffold(registry, W, H, ink_module, offsets)
    survey = all_conforming(sc, ptr, pair_signals(), registry, tolerance)
    conf = survey.pop("conforming")
    per = []
    fixed = {n.name: 0 for n in sc.nodes}
    for index in range(len(offsets)):
        sel = fixed | {"anchor": index}
        per.append({"anchor_bytes": offsets[index],
                    "train_accuracy": accuracy(sc, ptr, pair_signals(), registry, sel),
                    "held_accuracy": accuracy(sc, phe, pair_signals(), registry, sel)})
        report(f"T1 anchor {offsets[index]:>4} bytes  train / held",
               f"{per[-1]['train_accuracy']:.4f} / {per[-1]['held_accuracy']:.4f}")
    best = max(per, key=lambda r: r["train_accuracy"])
    result["t1"] = {"space_size": space_size(sc), "nodes": len(sc.nodes),
                    "train_records": len(ptr), "held_records": len(phe),
                    "conforming": survey | {"count": len(conf)},
                    "candidates": per, "best_by_train": best,
                    "always_different_baseline":
                        1 - sum(e["targets"]["same"].decoded for e in phe) / len(phe)}
    report("T1 space / evaluated / exhausted / conforming",
           f"{survey['space_size']} / {survey['evaluated']} / {survey['exhausted']} / {len(conf)}")
    report("T1 best anchor / held accuracy / always-different baseline",
           f"{best['anchor_bytes']} / {best['held_accuracy']:.4f} / "
           f"{result['t1']['always_different_baseline']:.4f}")
    dump(args.tag, result)


if __name__ == "__main__":
    main()
