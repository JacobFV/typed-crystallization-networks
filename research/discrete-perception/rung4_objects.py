"""Rung 4 -- `object_ids`: the multi-class rung, and where the ladder stops.

Three measurements, in the order that makes the verdict falsifiable.

1. **The information ceiling.**  A shared per-position module that reads only its
   own three bytes cannot beat the best RGB -> label lookup table.  That table is
   fitted on the training pixels and scored on held-out episodes, against the
   majority-class baseline.  This is a property of the data, not of a program,
   and it is computed outside the operator algebra deliberately: it is an upper
   bound on every program in that family.

2. **Exhaustive search in the colour-constant family** (the family rung 3 lives
   in), on the easiest possible reduction of the target -- the binary
   `object_ids == 0`.  `enumerate_fit` either finds a program or exhausts the
   space and certifies that none exists.

3. **Exhaustive search in the relational family.**  The only episode-adaptive
   handle the algebra has on an image is comparing a pixel to *another pixel*.
   The module's reference address is searched over every pixel in the image --
   the free-address choice the perception ladder measured relaxation to be worse
   than chance at, and which enumeration settles outright.  Run for the binary
   reduction and, at a smaller resolution, for the genuine multi-class target
   with an integer-valued module output.

Every family is also run on the rung-3 `foreground` target as a positive
control, so a "no solution" verdict is about the target and not the harness.
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import (BYTE, IDX, Builder, accuracy, bytes_type, dump, enumerate_reference,
                    episode, gradient_arm, pixel_starts, record_type, report, summarise)
from rung3_mask import POOL, module_scaffold, pixel_examples, signals as fg_signals
from tcn.graph import Signal
from tcn.operators import Registry
from tcn.search import space_size
from tcn.types import BOOL, Value, integer, product

CLASS = integer(8, signed=True)


# --------------------------------------------------------------------------
# 1.  information ceilings
# --------------------------------------------------------------------------

def label_sets(seeds, resolution, split, objects):
    """(rgb triple, labels) per pixel.  Probes are supervision only."""
    rows = []
    for s in seeds:
        pixels, probes = episode(s, resolution, split=split, objects=objects)
        ids = probes["object_ids"]
        depth = probes["depth"]
        n = resolution * resolution
        order = {}                          # canonical relabelling: raster-order rank
        for i in range(n):
            if ids[i] >= 0 and ids[i] not in order:
                order[ids[i]] = len(order)
        for i in range(n):
            rows.append((tuple(pixels[3 * i:3 * i + 3]),
                         {"foreground": ids[i] >= 0,
                          "object_ids": int(ids[i]),
                          "is_object_0": int(ids[i]) == 0,
                          "raster_rank": order.get(ids[i], -1) if ids[i] >= 0 else -1,
                          "depth_bin": 0 if depth[i] == 0 else 1 + min(3, int(depth[i]))}))
    return rows


def ceiling(train_rows, held_rows, key):
    table = collections.defaultdict(collections.Counter)
    for rgb, lab in train_rows:
        table[rgb][lab[key]] += 1
    best = {k: c.most_common(1)[0][0] for k, c in table.items()}
    majority = collections.Counter(lab[key] for _, lab in train_rows).most_common(1)[0][0]
    train_hit = sum(best[rgb] == lab[key] for rgb, lab in train_rows) / len(train_rows)
    hit = sum(best.get(rgb, majority) == lab[key] for rgb, lab in held_rows) / len(held_rows)
    maj = sum(majority == lab[key] for _, lab in held_rows) / len(held_rows)
    colliding = sum(1 for c in table.values() if len(c) > 1)
    return {"target": key, "distinct_rgb": len(table), "colliding_rgb": colliding,
            "train_accuracy": train_hit, "held_out_accuracy": hit,
            "majority_baseline": maj, "advantage_over_majority": hit - maj,
            "classes": len(set(lab[key] for _, lab in train_rows))}


def colour_multiplicity(seeds, resolution, objects, split="train"):
    """Distinct RGB values per object per image -- can equality separate instances?"""
    per_object, singleton = [], 0
    for s in seeds:
        pixels, probes = episode(s, resolution, split=split, objects=objects)
        ids = probes["object_ids"]
        groups = collections.defaultdict(set)
        for i in range(resolution * resolution):
            if ids[i] >= 0:
                groups[int(ids[i])].add(tuple(pixels[3 * i:3 * i + 3]))
        for k, colours in groups.items():
            per_object.append(len(colours))
            singleton += len(colours) == 1
    return {"objects_measured": len(per_object),
            "mean_distinct_colours_per_object": sum(per_object) / max(1, len(per_object)),
            "max_distinct_colours_per_object": max(per_object, default=0),
            "fraction_single_coloured": singleton / max(1, len(per_object))}


# --------------------------------------------------------------------------
# 2/3.  the searched families
# --------------------------------------------------------------------------

def examples_for(seeds, resolution, key, split="train", per_image=None, seed=0, objects=6,
                 value_type=BOOL):
    out = []
    REC = record_type(resolution)
    starts = pixel_starts(resolution)
    rng = random.Random(seed)
    for s in seeds:
        pixels, probes = episode(s, resolution, split=split, objects=objects)
        raw = Value.of(bytes_type(resolution), pixels).raw
        ids = probes["object_ids"]
        chosen = range(len(starts)) if per_image is None else sorted(
            rng.sample(range(len(starts)), min(per_image, len(starts))))
        for i in chosen:
            label = {"foreground": ids[i] >= 0, "is_object_0": int(ids[i]) == 0,
                     "object_ids": int(ids[i])}[key]
            out.append({"inputs": {"rec": Value(REC, (starts[i], raw))},
                        "targets": {key: Value.of(value_type, label)}})
    rng.shuffle(out)
    return out


def relational_scaffold(registry, resolution, target="is_object_0"):
    """Compare this pixel with the pixel at a *searched* reference address.

    |space| = (R*R) * 256: every reference address, and both Boolean combinators.
    """
    REC = record_type(resolution)
    starts = pixel_starts(resolution)
    consts = tuple((f"a{k}", Value.of(IDX, k)) for k in starts)
    consts += (("one", Value.of(IDX, 1)), ("two", Value.of(IDX, 2)))
    b = Builder(registry, (("rec", REC),), consts)
    b.add("pos", "project", ["rec"], params={"index": 0})
    b.add("obs", "project", ["rec"], params={"index": 1})
    b.add("g_at", "add", ["pos", "one"])
    b.add("b_at", "add", ["pos", "two"])
    b.add("red", "index", ["obs", "pos"])
    b.add("green", "index", ["obs", "g_at"])
    b.add("blue", "index", ["obs", "b_at"])
    b.choice("ref", [("identity", (f"a{k}",), None) for k in starts])
    b.add("ref_g", "add", ["ref", "one"])
    b.add("ref_b", "add", ["ref", "two"])
    b.add("r_ref", "index", ["obs", "ref"])
    b.add("g_ref", "index", ["obs", "ref_g"])
    b.add("b_ref", "index", ["obs", "ref_b"])
    b.add("same_r", "eq", ["red", "r_ref"])
    b.add("same_g", "eq", ["green", "g_ref"])
    b.add("same_b", "eq", ["blue", "b_ref"])
    b.choice("rg", [(f"truth_{t}", ("same_r", "same_g"), None) for t in range(16)])
    b.choice(target, [(f"truth_{t}", ("rg", "same_b"), None) for t in range(16)])
    b.add("record", "tuple", ["pos", target])
    return b.program((("y", "record"),))


def multiclass_scaffold(registry, resolution, classes=3):
    """A genuine multi-class head: `mux` over `classes` searched reference pixels.

    `cls = mux(same_as_ref0, 0, mux(same_as_ref1, 1, ... , -1))`, so the module's
    output is an integer class label, not a Boolean.  |space| = (R*R)^classes.
    """
    REC = record_type(resolution)
    starts = pixel_starts(resolution)
    consts = tuple((f"a{k}", Value.of(IDX, k)) for k in starts)
    consts += (("one", Value.of(IDX, 1)), ("two", Value.of(IDX, 2)))
    consts += tuple((f"c{j}", Value.of(CLASS, j)) for j in range(classes))
    consts += (("cneg", Value.of(CLASS, -1)),)
    b = Builder(registry, (("rec", REC),), consts)
    b.add("pos", "project", ["rec"], params={"index": 0})
    b.add("obs", "project", ["rec"], params={"index": 1})
    b.add("g_at", "add", ["pos", "one"])
    b.add("b_at", "add", ["pos", "two"])
    b.add("red", "index", ["obs", "pos"])
    b.add("green", "index", ["obs", "g_at"])
    b.add("blue", "index", ["obs", "b_at"])
    for j in range(classes):
        b.choice(f"ref{j}", [("identity", (f"a{k}",), None) for k in starts])
        b.add(f"ref{j}_g", "add", [f"ref{j}", "one"])
        b.add(f"ref{j}_b", "add", [f"ref{j}", "two"])
        b.add(f"r{j}", "index", ["obs", f"ref{j}"])
        b.add(f"g{j}", "index", ["obs", f"ref{j}_g"])
        b.add(f"b{j}", "index", ["obs", f"ref{j}_b"])
        b.add(f"eqr{j}", "eq", ["red", f"r{j}"])
        b.add(f"eqg{j}", "eq", ["green", f"g{j}"])
        b.add(f"eqb{j}", "eq", ["blue", f"b{j}"])
        b.add(f"and{j}", "and", [f"eqr{j}", f"eqg{j}"])
        b.add(f"same{j}", "and", [f"and{j}", f"eqb{j}"])
    prev = "cneg"
    for j in reversed(range(classes)):
        prev = b.add(f"sel{j}", "mux", [f"same{j}", f"c{j}", prev])
    b.add("cls", "identity", [prev])
    b.add("record", "tuple", ["pos", "cls"])
    return b.program((("y", "record"),))


def search(name, scaffold, train, held, sig, r, tolerance=1e-6, max_programs=1 << 24):
    enum = enumerate_reference(scaffold, train, sig, r, tolerance=tolerance,
                               max_programs=max_programs)
    enum["family"] = name
    if enum["solved"]:
        enum["held_out_accuracy"] = accuracy(scaffold, held, sig, r,
                                             selections=enum["selections"], tolerance=tolerance)
    report(f"{name:38s} space/solved/exhausted/unique/s",
           f"{enum['space_size']} / {enum['solved']} / {enum['exhausted']} / "
           f"{enum['unique']} / {enum['seconds']:.1f}")
    return enum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=8)
    ap.add_argument("--multiclass-resolution", type=int, default=6)
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--held", type=int, default=8)
    ap.add_argument("--per-image", type=int, default=48)
    ap.add_argument("--objects", type=int, default=6)
    ap.add_argument("--multiclass-objects", type=int, default=3)
    ap.add_argument("--gradient-seeds", type=int, default=4)
    ap.add_argument("--tag", default="rung4_objects")
    args = ap.parse_args()

    R = args.resolution
    train_seeds = tuple(range(args.train))
    held_seeds = tuple(range(100, 100 + args.held))
    result = {"arguments": vars(args)}
    tol = 1e-6

    # 1. ceilings
    tr = label_sets(train_seeds, R, "train", args.objects)
    hd = label_sets(held_seeds, R, "test", args.objects)
    result["ceilings"] = [ceiling(tr, hd, k) for k in
                          ("foreground", "object_ids", "is_object_0", "raster_rank", "depth_bin")]
    for c in result["ceilings"]:
        report(f"lookup ceiling {c['target']:12s} train/held/majority",
               f"{c['train_accuracy']:.4f} / {c['held_out_accuracy']:.4f} / "
               f"{c['majority_baseline']:.4f}  ({c['classes']} classes)")
    dump(args.tag, result)
    result["colour_multiplicity"] = colour_multiplicity(train_seeds, R, args.objects)
    report("distinct colours per object (mean/max/single)",
           f"{result['colour_multiplicity']['mean_distinct_colours_per_object']:.2f} / "
           f"{result['colour_multiplicity']['max_distinct_colours_per_object']} / "
           f"{result['colour_multiplicity']['fraction_single_coloured']:.3f}")

    r = Registry()
    result["families"] = []

    # 2. colour-constant family, on foreground (control) and on is_object_0
    for key, sig_name in (("foreground", "foreground"), ("is_object_0", "is_object_0")):
        train = examples_for(train_seeds, R, key, "train", args.per_image, seed=1,
                             objects=args.objects)
        held = examples_for(held_seeds, R, key, "test", args.per_image, seed=2,
                            objects=args.objects)
        scaffold = module_scaffold(r, R, POOL)
        sig = (Signal("foreground", key, ("core",), BOOL, "bce"),)
        result["families"].append(search(f"colour-constant / {key}", scaffold, train, held,
                                         sig, r, tolerance=tol))
        result["families"][-1]["positive_fraction"] = sum(
            ex["targets"][key].decoded for ex in train) / len(train)
        dump(args.tag, result)

    # 3. relational family, on foreground (control) and on is_object_0
    for key in ("foreground", "is_object_0"):
        train = examples_for(train_seeds, R, key, "train", args.per_image, seed=1,
                             objects=args.objects)
        held = examples_for(held_seeds, R, key, "test", args.per_image, seed=2,
                            objects=args.objects)
        scaffold = relational_scaffold(r, R, target="hit")
        sig = (Signal("hit", key, ("core",), BOOL, "bce"),)
        entry = search(f"relational / {key}", scaffold, train, held, sig, r, tolerance=tol)
        if key == "is_object_0":
            grad = gradient_arm(lambda: relational_scaffold(r, R, target="hit"),
                                tuple(range(args.gradient_seeds)), train, sig, r,
                                steps=300, lr=.15, init_noise=.5,
                                label="grad relational", tolerance=tol, held=held)
            entry["gradient"] = {"rows": grad, "summary": summarise(grad), "init_noise": .5}
        result["families"].append(entry)
        dump(args.tag, result)

    # 4. genuine multi-class, integer output, smaller image
    MR = args.multiclass_resolution
    train = examples_for(train_seeds, MR, "object_ids", "train", None, seed=1,
                         objects=args.multiclass_objects, value_type=CLASS)
    held = examples_for(held_seeds, MR, "object_ids", "test", None, seed=2,
                        objects=args.multiclass_objects, value_type=CLASS)
    scaffold = multiclass_scaffold(r, MR, classes=args.multiclass_objects)
    sig = (Signal("cls", "object_ids", ("core",), CLASS, "mse"),)
    entry = search(f"multi-class relational R{MR}", scaffold, train, held, sig, r, tolerance=tol)
    entry["classes"] = args.multiclass_objects
    entry["train_examples"] = len(train)
    entry["class_histogram"] = dict(collections.Counter(
        ex["targets"]["object_ids"].decoded for ex in train))
    result["families"].append(entry)

    dump(args.tag, result)


if __name__ == "__main__":
    main()
