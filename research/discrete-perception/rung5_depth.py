"""Rung 5 -- `depth`, and the one thing positional reuse adds that colour cannot.

`depth` is not a function of a pixel: the renderer shades by surface normal and
never attenuates with distance, so a colour carries object identity and facing,
not range.  What a shared per-position module *does* have that a per-pixel
colour rule does not is its own **position**, and position is numeric, so `lt`
and `le` are legal on it.  That makes "is this pixel in the near part of the
image" expressible, and it is the only depth-shaped thing in reach.

Measured here, for the binary target `near = (depth > 0) and (depth < T)`:

  * the information ceiling from **colour alone** and from **position alone**
    (best lookup table on each, scored on held-out episodes against majority);
  * `enumerate_fit` over the colour-constant family -- exhaustive, so a failure
    is a certificate rather than a budget;
  * a best-accuracy sweep over the **position-threshold** family and over the
    conjunction of both, because a target that no program fits exactly still has
    a best program, and the interesting number is how far above chance it gets.
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import random
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import (BYTE, IDX, Builder, accuracy, bytes_type, dump, enumerate_reference,
                    episode, gradient_arm, pixel_starts, record_type, report, summarise)
from rung3_mask import POOL, module_scaffold
from tcn.graph import Signal
from tcn.operators import Registry
from tcn.search import evaluate, space_size
from tcn.types import BOOL, Value


def threshold(seeds, resolution, objects):
    """Median foreground depth over the training episodes -- a declared constant."""
    xs = []
    for s in seeds:
        _, probes = episode(s, resolution, split="train", objects=objects)
        xs += [d for d in probes["depth"] if d > 0]
    return statistics.median(xs)


def rows(seeds, resolution, split, objects, T):
    out = []
    for s in seeds:
        pixels, probes = episode(s, resolution, split=split, objects=objects)
        d = probes["depth"]
        for i in range(resolution * resolution):
            out.append({"index": i, "rgb": tuple(pixels[3 * i:3 * i + 3]),
                        "near": bool(d[i] > 0 and d[i] < T),
                        "depth": d[i]})
    return out


def lookup_ceiling(train, held, feature, key="near"):
    table = collections.defaultdict(collections.Counter)
    for row in train:
        table[feature(row)][row[key]] += 1
    best = {k: c.most_common(1)[0][0] for k, c in table.items()}
    majority = collections.Counter(r[key] for r in train).most_common(1)[0][0]
    return {"distinct_keys": len(table),
            "colliding_keys": sum(1 for c in table.values() if len(c) > 1),
            "train_accuracy": sum(best[feature(r)] == r[key] for r in train) / len(train),
            "held_out_accuracy": sum(best.get(feature(r), majority) == r[key] for r in held) / len(held),
            "majority_baseline": sum(majority == r[key] for r in held) / len(held)}


def examples(seeds, resolution, split, objects, T, per_image=None, seed=0):
    out = []
    REC = record_type(resolution)
    starts = pixel_starts(resolution)
    rng = random.Random(seed)
    for s in seeds:
        pixels, probes = episode(s, resolution, split=split, objects=objects)
        raw = Value.of(bytes_type(resolution), pixels).raw
        d = probes["depth"]
        chosen = range(len(starts)) if per_image is None else sorted(
            rng.sample(range(len(starts)), min(per_image, len(starts))))
        for i in chosen:
            out.append({"inputs": {"rec": Value(REC, (starts[i], raw))},
                        "targets": {"near": Value.of(BOOL, bool(d[i] > 0 and d[i] < T))}})
    rng.shuffle(out)
    return out


def position_scaffold(registry, resolution, pool=POOL, cuts=16):
    """`near = truth_T( lt/ge(pos, cut), foreground_test )`.

    The position threshold is a genuine numeric comparison -- legal because a
    position is an ordinary integer, unlike a pixel byte.
    """
    REC = record_type(resolution)
    n = resolution * resolution
    step = max(1, n // cuts)
    limits = tuple(3 * k for k in range(step, n, step))
    consts = tuple((f"cut{v}", Value.of(IDX, v)) for v in limits)
    consts += tuple((f"byte_{v}", Value.of(BYTE, v)) for v in pool)
    consts += (("one", Value.of(IDX, 1)), ("two", Value.of(IDX, 2)))
    b = Builder(registry, (("rec", REC),), consts)
    b.add("pos", "project", ["rec"], params={"index": 0})
    b.add("obs", "project", ["rec"], params={"index": 1})
    b.add("g_at", "add", ["pos", "one"])
    b.add("b_at", "add", ["pos", "two"])
    b.add("red", "index", ["obs", "pos"])
    b.add("green", "index", ["obs", "g_at"])
    b.add("blue", "index", ["obs", "b_at"])
    for name, source in (("cmp_r", "red"), ("cmp_g", "green"), ("cmp_b", "blue")):
        b.choice(name, [("eq", (source, f"byte_{v}"), None) for v in pool])
    b.choice("rg", [(f"truth_{t}", ("cmp_r", "cmp_g"), None) for t in range(16)])
    b.choice("fg", [(f"truth_{t}", ("rg", "cmp_b"), None) for t in range(16)])
    b.choice("side", [(op, ("pos", f"cut{v}"), None) for v in limits for op in ("lt", "ge")])
    b.choice("near", [(f"truth_{t}", ("fg", "side"), None) for t in range(16)])
    b.add("record", "tuple", ["pos", "near"])
    return b.program((("y", "record"),)), limits


def best_in_family(program, train, held, sig, registry, cap=None):
    """Best *accuracy* program in the family, not best conforming program.

    A target no program fits exactly still has a best program, and the useful
    number is how far above the majority baseline it gets on held-out episodes.
    Every candidate is executed through `Program.execute`, the same path
    `enumerate_fit` uses; only the objective differs.
    """
    import itertools
    names = [n.name for n in program.nodes]
    counts = [range(len(n.candidates)) for n in program.nodes]
    t0 = time.perf_counter()
    best, best_sel, evaluated = -1., None, 0
    for combination in itertools.product(*counts):
        if cap is not None and evaluated >= cap:
            break
        evaluated += 1
        sel = dict(zip(names, combination))
        a = accuracy(program, train, sig, registry, selections=sel)
        if a > best:
            best, best_sel = a, sel
    return {"evaluated": evaluated, "space_size": space_size(program),
            "exhausted": cap is None or evaluated >= space_size(program),
            "best_train_accuracy": best, "selections": best_sel,
            "held_out_accuracy": accuracy(program, held, sig, registry, selections=best_sel),
            "seconds": time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=8)
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--held", type=int, default=8)
    ap.add_argument("--per-image", type=int, default=32)
    ap.add_argument("--objects", type=int, default=6)
    ap.add_argument("--cuts", type=int, default=16)
    ap.add_argument("--joint-budget", type=int, default=100000)
    ap.add_argument("--gradient-seeds", type=int, default=4)
    ap.add_argument("--tag", default="rung5_depth")
    args = ap.parse_args()

    R = args.resolution
    train_seeds = tuple(range(args.train))
    held_seeds = tuple(range(100, 100 + args.held))
    tol = 1e-6
    T = threshold(train_seeds, R, args.objects)
    result = {"arguments": vars(args), "depth_threshold": T}
    report("median foreground depth (the declared threshold)", f"{T:.4f}")

    tr = rows(train_seeds, R, "train", args.objects, T)
    hd = rows(held_seeds, R, "test", args.objects, T)
    result["near_fraction_train"] = sum(r["near"] for r in tr) / len(tr)
    result["ceiling_colour"] = lookup_ceiling(tr, hd, lambda r: r["rgb"])
    result["ceiling_position"] = lookup_ceiling(tr, hd, lambda r: r["index"])
    result["ceiling_colour_and_position"] = lookup_ceiling(tr, hd, lambda r: (r["rgb"], r["index"]))
    dump(args.tag, result)
    for k in ("ceiling_colour", "ceiling_position", "ceiling_colour_and_position"):
        c = result[k]
        report(f"{k:32s} train/held/majority",
               f"{c['train_accuracy']:.4f} / {c['held_out_accuracy']:.4f} / "
               f"{c['majority_baseline']:.4f}")

    r = Registry()
    train = examples(train_seeds, R, "train", args.objects, T, args.per_image, seed=1)
    held = examples(held_seeds, R, "test", args.objects, T, args.per_image, seed=2)
    result["train_examples"] = len(train)

    # colour-constant family: exhaustive conformance search -> a certificate
    colour = module_scaffold(r, R, POOL)
    sig_colour = (Signal("foreground", "near", ("core",), BOOL, "bce"),)
    enum = enumerate_reference(colour, train, sig_colour, r, tolerance=tol)
    result["colour_family"] = enum
    dump(args.tag, result)
    report("colour-constant family space/solved/exhausted/seconds",
           f"{enum['space_size']} / {enum['solved']} / {enum['exhausted']} / {enum['seconds']:.1f}")

    # position + colour family: too large to exhaust; measure the rate and project
    pos_prog, limits = position_scaffold(r, R, POOL, args.cuts)
    sig = (Signal("near", "near", ("core",), BOOL, "bce"),)
    result["position_family_cuts"] = list(limits)
    enum2 = enumerate_reference(pos_prog, train, sig, r, tolerance=tol,
                                max_programs=args.joint_budget)
    enum2["programs_per_second"] = enum2["evaluated"] / max(1e-9, enum2["seconds"])
    enum2["projected_exhaustive_seconds"] = enum2["space_size"] / max(1e-9, enum2["programs_per_second"])
    result["position_family"] = enum2
    dump(args.tag, result)
    report("position+colour family space/evaluated/exhausted/seconds",
           f"{enum2['space_size']} / {enum2['evaluated']} / {enum2['exhausted']} / "
           f"{enum2['seconds']:.1f}")
    report("position+colour projected exhaustive wall clock (s)",
           f"{enum2['projected_exhaustive_seconds']:.3g}")

    # The same family with the colour comparisons pinned to the (already
    # searched, already certified) background test: 480 programs, exhaustible,
    # and it isolates what position alone adds.
    from dataclasses import replace
    fixed = {"cmp_r": POOL.index(24), "cmp_g": POOL.index(30), "cmp_b": POOL.index(43),
             "rg": 8, "fg": 7}
    nodes = [n if n.name in ("side", "near") else
             replace(n, candidates=(n.candidates[fixed.get(n.name, 0)],), selected=0)
             for n in pos_prog.nodes]
    reduced = replace(pos_prog, nodes=tuple(nodes)).validate(r)
    enum3 = enumerate_reference(reduced, train, sig, r, tolerance=tol)
    result["position_only_conformance"] = enum3
    dump(args.tag, result)
    report("position-only family space/solved/exhausted/seconds",
           f"{enum3['space_size']} / {enum3['solved']} / {enum3['exhausted']} / "
           f"{enum3['seconds']:.1f}")
    best = best_in_family(reduced, train, held, sig, r)
    result["position_only_best"] = best
    dump(args.tag, result)
    report("position-only best train/held-out accuracy",
           f"{best['best_train_accuracy']:.4f} / {best['held_out_accuracy']:.4f} "
           f"({best['space_size']} programs, {best['seconds']:.1f}s)")

    grad = gradient_arm(lambda: position_scaffold(r, R, POOL, args.cuts)[0],
                        tuple(range(args.gradient_seeds)), train, sig, r,
                        steps=300, lr=.15, init_noise=.5, label="grad depth",
                        tolerance=tol, held=held)
    result["gradient"] = {"rows": grad, "summary": summarise(grad), "init_noise": .5}
    report("gradient successes on the position+colour family",
           f"{result['gradient']['summary']['successes']}/{len(grad)}")

    dump(args.tag, result)


if __name__ == "__main__":
    main()
