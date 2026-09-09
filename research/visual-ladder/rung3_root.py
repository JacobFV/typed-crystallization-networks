"""H6 repaired: the root is recovered by the program, not supplied to the scorer.

`rung3_widgets.py` reads a left and an upper neighbour, so position (0, 0) is
outside the position set by construction and 12 of 227 widgets are never
predicted.  `root_rule.py` first checks the repair as a *rule* at every instance
-- an off-screen neighbour counts as "different" -- and finds 227 true positives,
0 false, 0 missed over 12,288 positions on 12 held-out screens.  This searches
for the program.

What is added, and it is added structure (H1), so it is declared:

* S1' masks each `same` call with an in-bounds test built from the *searched*
  offset, not from a supplied one: the neighbour exists (`pos >= off`, with the
  address clamped by `max` so it cannot go below the origin) and did not wrap
  into the previous row (`col(pos - off) <= col(pos)`, `col = mod stride`).  The
  masked signal is "there is a neighbour there and it is the same colour", so an
  off-screen neighbour reads as different and (0, 0) becomes a corner.  The
  offset pool, the pair of offsets and the 16-entry truth table are searched
  exactly as before: the space is still 5 * 5 * 16 = 400.
* S2' clamps its parent pixel the same way, `left = max(pos, 3) - 3`.  At the
  origin that reads the widget's *own* pixel, so `parent_key == own_key` is the
  program's own root marker -- the widget with no containing rectangle -- rather
  than a scorer convention.  The step pool and space (25) are unchanged.
* the position set is every pixel, 1024 rather than 961, and the supervision
  includes the root, so nothing about the root is special-cased downstream.

S0 is reused frozen from `out/rung3.json`; it is unchanged by any of this.
The parse is scored with the position re-key of `rekey.py`.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import (FLAT, IDX, Builder, Registry, bytes_type, colour_at, dump, episode,
                    load, record_type, report)
from tcn.types import BOOL, Value

import rung3_widgets as R
from rekey import score, totals


# --------------------------------------------------------------------------
def all_positions(ep):
    """Every pixel address, the root's corner included: H6 lifted."""
    w, h = ep["width"], ep["height"]
    return tuple(3 * (y * w + x) for y in range(h) for x in range(w))


def corner_scaffold_masked(registry, width, height, module, offsets):
    """`rec -> bool`, with an off-screen neighbour reading as "different"."""
    consts = tuple((f"off{k}", Value.of(IDX, k)) for k in offsets)
    consts += (("stride", Value.of(IDX, 3 * width)),)
    b = Builder(registry, (("rec", record_type(width, height)),), consts)
    b.add("pos", "project", ["rec"], params={"index": 0})
    b.add("obs", "project", ["rec"], params={"index": 1})
    b.add("col", "mod", ["pos", "stride"])
    for arm in ("a", "b"):
        b.choice(f"back_{arm}", [("identity", (f"off{k}",), None, None) for k in offsets])
        b.add(f"safe_{arm}", "max", ["pos", f"back_{arm}"])
        b.add(f"addr_{arm}", "sub", [f"safe_{arm}", f"back_{arm}"])
        b.add(f"exists_{arm}", "ge", ["pos", f"back_{arm}"])
        b.add(f"bcol_{arm}", "mod", [f"addr_{arm}", "stride"])
        b.add(f"norap_{arm}", "le", [f"bcol_{arm}", "col"])
        b.add(f"inb_{arm}", "and", [f"exists_{arm}", f"norap_{arm}"])
        b.add(f"raw_{arm}", module, [f"addr_{arm}", "pos", "obs"])
        b.add(f"same_{arm}", "and", [f"raw_{arm}", f"inb_{arm}"])
    b.choice("corner", [(f"truth_{t}", ("same_a", "same_b"), None, None) for t in range(16)])
    return b.program((("y", "corner"),))


def corner_examples_all(seeds, split, per_image=None, seed=0, **configuration):
    import random
    rng = random.Random(seed)
    rows = []
    for s in seeds:
        ep = episode(s, split, **configuration)
        REC = record_type(ep["width"], ep["height"])
        raw = Value.of(bytes_type(ep["width"], ep["height"]), ep["pixels"]).raw
        corners = {(d["rect"][0], d["rect"][1]) for d in ep["probes"]["hierarchy"]}
        positions = list(all_positions(ep))
        if per_image is not None and per_image < len(positions):
            keep = sorted(rng.sample(range(len(positions)), per_image))
            # the root is one position in 1024; a uniform sample would usually
            # miss it, and a stage never shown the case it was added for is not
            # a test of it.  Its corner is forced in and the count is declared.
            keep = sorted(set(keep) | {0})
            positions = [positions[i] for i in keep]
        for p in positions:
            i = p // 3
            x, y = i % ep["width"], i // ep["width"]
            rows.append({"inputs": {"rec": Value(REC, (p, raw))},
                         "targets": {"corner": Value.of(BOOL, (x, y) in corners)}})
    rng.shuffle(rows)
    return rows


def rect_scaffold_clamped(registry, width, height, module, offsets, span=None):
    """`rect_scaffold` with the parent pixel clamped at the origin."""
    # rebuilt rather than patched: the only difference is the clamped `left`.
    span = span or max(width, height)
    last = 3 * width * height - 3
    consts = (("one", Value.of(IDX, 1)), ("two", Value.of(IDX, 2)),
              ("three", Value.of(IDX, 3)), ("last", Value.of(IDX, last)),
              ("width", Value.of(IDX, width)), ("height", Value.of(IDX, height)))
    consts += tuple((f"off{k}", Value.of(IDX, k)) for k in offsets)
    consts += tuple((f"k{k}", Value.of(IDX, k)) for k in range(1, span))
    b = Builder(registry, (("rec", record_type(width, height)),), consts)
    b.add("pos", "project", ["rec"], params={"index": 0})
    b.add("obs", "project", ["rec"], params={"index": 1})
    b.add("linear", "idiv", ["pos", "three"])
    b.add("x", "mod", ["linear", "width"])
    b.add("y", "idiv", ["linear", "width"])
    b.choice("step_w", [("identity", (f"off{k}",), None, None) for k in offsets])
    b.choice("step_h", [("identity", (f"off{k}",), None, None) for k in offsets])
    for tag, step, coordinate, limit in (("w", "step_w", "x", "width"),
                                         ("h", "step_h", "y", "height")):
        terms, run = [], None
        for k in range(1, span):
            b.add(f"{tag}d{k}", "mul", [step, f"k{k}"])
            b.add(f"{tag}a{k}", "add", ["pos", f"{tag}d{k}"])
            b.add(f"{tag}c{k}", "min", [f"{tag}a{k}", "last"])
            b.add(f"{tag}s{k}", module, [f"{tag}c{k}", "pos", "obs"])
            b.add(f"{tag}i{k}", "add", [coordinate, f"k{k}"])
            b.add(f"{tag}m{k}", "lt", [f"{tag}i{k}", limit])
            b.add(f"{tag}t{k}", "and", [f"{tag}s{k}", f"{tag}m{k}"])
            run = (f"{tag}t{k}" if run is None
                   else b.add(f"{tag}r{k}", "and", [run, f"{tag}t{k}"]))
            terms.append(b.add(f"{tag}e{k}", "encode", [run], out=IDX))
        b.add(f"{tag}_tuple", "tuple", terms)
        b.add(f"{tag}_count", "sum", [f"{tag}_tuple"])
        b.add(f"{tag}_extent", "add", [f"{tag}_count", "one"])
    b.add("safe_left", "max", ["pos", "three"])
    b.add("left", "sub", ["safe_left", "three"])
    for tag, base in (("own", "pos"), ("par", "left")):
        b.add(f"{tag}_g", "add", [base, "one"])
        b.add(f"{tag}_b", "add", [base, "two"])
        b.add(f"{tag}_rv", "index", ["obs", base])
        b.add(f"{tag}_gv", "index", ["obs", f"{tag}_g"])
        b.add(f"{tag}_bv", "index", ["obs", f"{tag}_b"])
        b.add(f"{tag}_tup", "tuple", [f"{tag}_rv", f"{tag}_gv", f"{tag}_bv"])
        b.add(f"{tag}_key", "pack", [f"{tag}_tup"], out=R.KEY)
    b.add("record", "tuple", ["x", "y", "w_extent", "h_extent", "own_key", "par_key"])
    return b.program((("y", "record"),))


def rect_examples_all(seeds, split, **configuration):
    """One example per widget, the root included; its parent pixel is itself."""
    rows = []
    for s in seeds:
        ep = episode(s, split, **configuration)
        REC = record_type(ep["width"], ep["height"])
        raw = Value.of(bytes_type(ep["width"], ep["height"]), ep["pixels"]).raw
        for d in ep["probes"]["hierarchy"]:
            x, y = d["rect"][0], d["rect"][1]
            own = R.pack_rgb(colour_at(ep, x, y))
            par = own if x == 0 else R.pack_rgb(colour_at(ep, x - 1, y))
            target = Value.of(R.RECT, (x, y, d["rect"][2], d["rect"][3], own, par))
            rows.append({"inputs": {"rec": Value(REC, (3 * (y * ep["width"] + x), raw))},
                         "targets": {"rect": target}})
    return rows


# --------------------------------------------------------------------------
def score_with_root(predicted, ep):
    """Position re-key, with the root produced by the program rather than supplied.

    A row is the root when its `parent_key` equals its `own_key` -- which the
    clamp makes true exactly at the origin, and which cannot happen elsewhere
    because the corner predicate already says the pixel to the left differs.
    Every other row's parent is the smallest parsed rectangle containing the
    pixel at `(x - 1, y)`, an address its own corner determines.
    """
    truth = {d["id"]: d for d in ep["probes"]["hierarchy"]}
    rects = {i: tuple(d["rect"]) for i, d in truth.items()}
    got_rects = {tuple(r[:4]) for r in predicted}
    want_rects = {tuple(d["rect"]) for d in truth.values()}
    right = wrong = 0
    for r in predicted:
        if r[5] == r[4]:
            got = None
        else:
            inside = [q for q in predicted
                      if q[0] <= r[0] - 1 < q[0] + q[2] and q[1] <= r[1] < q[1] + q[3]]
            got = tuple(min(inside, key=lambda q: q[2] * q[3])[:4]) if inside else None
        match = [d for d in truth.values() if tuple(d["rect"]) == tuple(r[:4])]
        want = None if (len(match) == 1 and match[0]["parent"] == 255) else \
            (tuple(rects[match[0]["parent"]]) if len(match) == 1 else "?")
        if len(match) == 1 and got == want:
            right += 1
        else:
            wrong += 1
    return {"widgets_in_probe": len(truth),
            "rects_predicted": len(got_rects), "rects_true": len(want_rects),
            "rects_exact": got_rects == want_rects,
            "rects_missing": sorted(want_rects - got_rects),
            "rects_spurious": sorted(got_rects - want_rects),
            "roots_predicted": sum(1 for r in predicted if r[5] == r[4]),
            "corner_key_collisions": len(predicted) - len({(r[0], r[1]) for r in predicted}),
            "parent_links_correct": right, "parent_links_wrong": wrong,
            "tree_exact": got_rects == want_rects and wrong == 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="rung3")
    ap.add_argument("--tag", default="rung3_root")
    ap.add_argument("--train", type=int, default=6)
    ap.add_argument("--validation", type=int, default=3)
    ap.add_argument("--held", type=int, default=6)
    ap.add_argument("--corner-per-image", type=int, default=120)
    ap.add_argument("--first", type=int, default=200)
    ap.add_argument("--screens", type=int, default=12)
    args = ap.parse_args()

    registry = Registry()
    tolerance = 1e-6
    configuration = dict(FLAT)
    found = load(args.source)
    probe = episode(0, "train", **configuration)
    W, H = probe["width"], probe["height"]
    offsets = R.offset_pool(W)
    p0 = R.same_scaffold(registry, W, H)
    same_module = registry.register_module(p0.harden(found["s0"]["chosen"]))
    result = {"arguments": vars(args), "configuration": configuration,
              "s0_reused_from": args.source, "s0_selections": found["s0"]["chosen"],
              "positions": len(all_positions(probe)),
              "positions_before": len(R.interior_positions(probe)),
              "offset_pool": list(offsets)}

    print("\n--- S1': corner with an in-bounds mask, so the origin is a corner ---")
    ctr = corner_examples_all(range(args.train), "train", args.corner_per_image, seed=4,
                              **configuration)
    cva = corner_examples_all(range(50, 50 + args.validation), "validation",
                              args.corner_per_image, seed=5, **configuration)
    che = corner_examples_all(range(100, 100 + args.held), "test", args.corner_per_image,
                              seed=6, **configuration)
    result["s1_positive_fraction"] = sum(e["targets"]["corner"].decoded for e in ctr) / len(ctr)
    corner_program = corner_scaffold_masked(registry, W, H, same_module, offsets)
    result["s1"], _ = stage_or_die("S1' corner masked", corner_program, ctr, cva, che,
                                   R.corner_signals(), registry, tolerance)
    corner_module = registry.register_module(corner_program.harden(result["s1"]["chosen"]))
    result["s1"]["offsets_chosen"] = [offsets[result["s1"]["chosen"]["back_a"]],
                                      offsets[result["s1"]["chosen"]["back_b"]]]
    report("S1' offsets chosen (bytes)", result["s1"]["offsets_chosen"])
    dump(args.tag, result)

    print("\n--- S2': the rectangle, with the parent pixel clamped at the origin ---")
    rtr = rect_examples_all(range(args.train), "train", **configuration)
    rva = rect_examples_all(range(50, 50 + args.validation), "validation", **configuration)
    rhe = rect_examples_all(range(100, 100 + args.held), "test", **configuration)
    rect_program = rect_scaffold_clamped(registry, W, H, same_module, offsets)
    result["s2"], _ = stage_or_die("S2' rect clamped", rect_program, rtr, rva, rhe,
                                   R.rect_signals(), registry, tolerance, draws=50)
    rect_module = registry.register_module(rect_program.harden(result["s2"]["chosen"]))
    result["s2"]["steps_chosen"] = [offsets[result["s2"]["chosen"]["step_w"]],
                                    offsets[result["s2"]["chosen"]["step_h"]]]
    report("S2' steps chosen (bytes)", result["s2"]["steps_chosen"])
    dump(args.tag, result)

    print("\n--- S3': the whole screen, root included, position re-key ---")
    rows = []
    for seed in range(args.first, args.first + args.screens):
        ep = episode(seed, "test", **configuration)
        BT = bytes_type(ep["width"], ep["height"])
        positions = all_positions(ep)
        parser = R.assembly(registry, BT, positions, corner_module, rect_module)
        started = time.perf_counter()
        got = parser.run({"observation": Value.of(BT, ep["pixels"])}, registry=registry)[0]
        predicted = [list(map(int, r)) for r in sorted(got["mapped"].decoded)]
        row = score_with_root(predicted, ep) | {"seed": seed,
                                                "seconds": time.perf_counter() - started,
                                                "positions": len(positions)}
        rows.append(row)
        report(f"parse seed {seed}", f"rects {row['rects_predicted']}/{row['rects_true']} "
               f"exact={row['rects_exact']} roots {row['roots_predicted']} links "
               f"{row['parent_links_correct']}/{row['widgets_in_probe']} "
               f"tree_exact={row['tree_exact']}")
    total = {"screens": len(rows),
             "widgets_in_probe": sum(r["widgets_in_probe"] for r in rows),
             "rects_predicted": sum(r["rects_predicted"] for r in rows),
             "screens_rects_exact": sum(r["rects_exact"] for r in rows),
             "roots_predicted": sum(r["roots_predicted"] for r in rows),
             "parent_links_correct": sum(r["parent_links_correct"] for r in rows),
             "parent_links_wrong": sum(r["parent_links_wrong"] for r in rows),
             "corner_key_collisions": sum(r["corner_key_collisions"] for r in rows),
             "trees_exact": sum(r["tree_exact"] for r in rows)}
    total["rect_recall"] = total["rects_predicted"] / max(1, total["widgets_in_probe"])
    total["link_accuracy"] = total["parent_links_correct"] / max(1, total["widgets_in_probe"])
    result["parse"] = {"episodes": rows, "totals": total}
    report("S3' totals", total)
    dump(args.tag, result)


def stage_or_die(name, *rest, **kw):
    out, conforming = R.stage(name, *rest, **kw)
    if "chosen" not in out:
        raise SystemExit(f"{name}: 0 conforming, exhausted -- that is the certificate.")
    return out, conforming


if __name__ == "__main__":
    main()
