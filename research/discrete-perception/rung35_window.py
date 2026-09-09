"""Rung 3.5 -- a learned two-position spatial operator, by staged composition.

The perception ladder's operator audit says an image has "no convolution, no
window, no pooling": every program that reads pixels must name individual bytes
by absolute index, and there is no aggregation of any kind.

Both halves of that stop being true once the image is a *set of positional
records*.  A shared module receives its own position and can address relative to
it with `add`, which is a window; and `filter`+`count` over the mapped set is an
aggregate.  Neither adds an operator.

The rung: predict whether the foreground mask *changes* between raster position
`i` and position `i + offset` -- a two-pixel edge detector, with the offset
itself searched.  Measured two ways over the identical target:

* **staged**: the rung-3 foreground module is frozen and registered, and the
  window module is searched over `offset x combinator` only.  Two stages, each
  with its own probe (`foreground` then `edge`).
* **flat**: the same predicate searched in one scaffold, with both pixels' three
  channel comparisons and all five combinators free.  This is the same space
  without the decomposition, and it is where brute force stops.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import (BACKGROUND, BYTE, IDX, Builder, accuracy, all_conforming, bytes_type,
                    dump, enumerate_reference, episode, exact_error, gradient_arm,
                    pixel_starts, positional_caller, random_reference, record_type, report,
                    summarise)
from rung3_mask import POOL, module_scaffold, pixel_examples, signals as fg_signals
from tcn.graph import Signal
from tcn.operators import Registry
from tcn.search import evaluate, space_size
from tcn.types import BOOL, Value, product, setof


def offsets(resolution):
    """Relative addresses the window may look at: right, two right, one row down."""
    return (3, 6, 3 * resolution)


def window_positions(resolution):
    """Positions at which every candidate offset stays inside the image."""
    n = resolution * resolution
    return tuple(3 * i for i in range(n - resolution))


def edge_labels(probes, resolution, offset=3):
    """`fg(i) != fg(i + offset/3)` -- the target, from the `object_ids` probe."""
    ids = probes["object_ids"]
    step = offset // 3
    return {3 * i: (ids[i] >= 0) != (ids[i + step] >= 0)
            for i in range(resolution * resolution - resolution)}


def window_examples(seeds, resolution, split="train", per_image=None, seed=0, objects=6):
    out = []
    REC = record_type(resolution)
    rng = random.Random(seed)
    positions = window_positions(resolution)
    for s in seeds:
        pixels, probes = episode(s, resolution, split=split, objects=objects)
        raw = Value.of(bytes_type(resolution), pixels).raw
        labels = edge_labels(probes, resolution)
        chosen = positions if per_image is None else [
            positions[i] for i in sorted(rng.sample(range(len(positions)),
                                                    min(per_image, len(positions))))]
        for start in chosen:
            out.append({"inputs": {"rec": Value(REC, (start, raw))},
                        "targets": {"edge": Value.of(BOOL, labels[start])}})
    rng.shuffle(out)
    return out


def edge_signals():
    return (Signal("edge", "edge", ("core",), BOOL, "bce"),)


def staged_scaffold(registry, resolution, module_name):
    """Calls one frozen foreground module twice: |offsets| x 16 programs."""
    REC = record_type(resolution)
    consts = tuple((f"off{k}", Value.of(IDX, k)) for k in offsets(resolution))
    b = Builder(registry, (("rec", REC),), consts)
    b.add("pos", "project", ["rec"], params={"index": 0})
    b.add("obs", "project", ["rec"], params={"index": 1})
    b.add("here_rec", "tuple", ["pos", "obs"])
    b.add("here", module_name, ["here_rec"])
    b.add("here_flag", "project", ["here"], params={"index": 1})
    b.choice("shifted", [("add", ("pos", f"off{k}"), None) for k in offsets(resolution)])
    b.add("there_rec", "tuple", ["shifted", "obs"])
    b.add("there", module_name, ["there_rec"])
    b.add("there_flag", "project", ["there"], params={"index": 1})
    b.choice("edge", [(f"truth_{t}", ("here_flag", "there_flag"), None) for t in range(16)])
    b.add("record", "tuple", ["pos", "edge"])
    return b.program((("y", "record"),))


def flat_scaffold(registry, resolution, pool=POOL):
    """The same predicate with nothing frozen: |pool|^6 * 16^5 programs."""
    REC = record_type(resolution)
    consts = tuple((f"byte_{v}", Value.of(BYTE, v)) for v in pool)
    consts += tuple((f"off{k}", Value.of(IDX, k)) for k in (1, 2))
    consts += tuple((f"step{k}", Value.of(IDX, k)) for k in offsets(resolution))
    b = Builder(registry, (("rec", REC),), consts)
    b.add("pos", "project", ["rec"], params={"index": 0})
    b.add("obs", "project", ["rec"], params={"index": 1})
    b.choice("shifted", [("add", ("pos", f"step{k}"), None) for k in offsets(resolution)])
    for tag, base in (("a", "pos"), ("b", "shifted")):
        b.add(f"{tag}_g", "add", [base, "off1"])
        b.add(f"{tag}_b", "add", [base, "off2"])
        b.add(f"{tag}_red", "index", ["obs", base])
        b.add(f"{tag}_green", "index", ["obs", f"{tag}_g"])
        b.add(f"{tag}_blue", "index", ["obs", f"{tag}_b"])
        for ch in ("red", "green", "blue"):
            b.choice(f"{tag}_cmp_{ch}", [("eq", (f"{tag}_{ch}", f"byte_{v}"), None) for v in pool])
        b.choice(f"{tag}_rg", [(f"truth_{t}", (f"{tag}_cmp_red", f"{tag}_cmp_green"), None)
                               for t in range(16)])
        b.choice(f"{tag}_fg", [(f"truth_{t}", (f"{tag}_rg", f"{tag}_cmp_blue"), None)
                               for t in range(16)])
    b.choice("edge", [(f"truth_{t}", ("a_fg", "b_fg"), None) for t in range(16)])
    b.add("record", "tuple", ["pos", "edge"])
    return b.program((("y", "record"),))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=8)
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--held", type=int, default=8)
    ap.add_argument("--per-image", type=int, default=48)
    ap.add_argument("--apply", type=int, default=3)
    ap.add_argument("--gradient-seeds", type=int, default=8)
    ap.add_argument("--flat-budget", type=int, default=200000)
    ap.add_argument("--objects", type=int, default=6)
    ap.add_argument("--tag", default="rung35_window")
    args = ap.parse_args()

    R = args.resolution
    if args.per_image is not None and args.per_image <= 0:
        args.per_image = None          # supervise every position
    r = Registry()
    tol = 1e-6
    train_seeds = tuple(range(args.train))
    held_seeds = tuple(range(100, 100 + args.held))
    result = {"arguments": vars(args), "resolution": R, "offsets": list(offsets(R)),
              "window_positions": len(window_positions(R))}

    # ---- stage 1: the rung-3 foreground module, searched exhaustively ----
    fg_train = pixel_examples(train_seeds, R, "train", args.per_image, seed=1, objects=args.objects)
    fg_held = pixel_examples(held_seeds, R, "test", args.per_image, seed=2, objects=args.objects)
    fg_scaffold = module_scaffold(r, R, POOL)
    fg_sig = fg_signals()
    from incremental import incremental_conforming
    fg_enum = incremental_conforming(fg_scaffold, fg_train, fg_sig, r, tolerance=tol)
    fg_enum["solved"] = fg_enum["count"] > 0
    fg_enum["selections"] = fg_enum["conforming"][0] if fg_enum["conforming"] else None
    fg_enum.pop("conforming", None)
    fg_enum["held_out_max_error"] = exact_error(fg_scaffold, fg_held, fg_sig, r,
                                                selections=fg_enum["selections"])
    result["stage1_foreground"] = fg_enum
    dump(args.tag, result)
    report("stage 1 space/exhausted/unique/seconds/held-out error",
           f"{fg_enum['space_size']} / {fg_enum['exhausted']} / {fg_enum['unique']} / "
           f"{fg_enum['seconds']:.1f} / {fg_enum['held_out_max_error']}")
    # The staged composition is only as good as the frozen stage.  Measure the
    # stage-1 module at EVERY position, not only the supervised sample.
    all_train = pixel_examples(train_seeds, R, "train", None, seed=11, objects=args.objects)
    all_held = pixel_examples(held_seeds, R, "test", None, seed=12, objects=args.objects)
    fg_enum["accuracy_all_positions_train"] = accuracy(
        fg_scaffold, all_train, fg_sig, r, selections=fg_enum["selections"])
    fg_enum["accuracy_all_positions_held_out"] = accuracy(
        fg_scaffold, all_held, fg_sig, r, selections=fg_enum["selections"])
    report("stage 1 module accuracy at every position, train / held-out",
           f"{fg_enum['accuracy_all_positions_train']:.6f} / "
           f"{fg_enum['accuracy_all_positions_held_out']:.6f}")
    fg_module = fg_scaffold.harden(fg_enum["selections"])
    name = r.register_module(fg_module)

    # ---- stage 2: the window, staged on the frozen module ----
    train = window_examples(train_seeds, R, "train", args.per_image, seed=3, objects=args.objects)
    held = window_examples(held_seeds, R, "test", args.per_image, seed=4, objects=args.objects)
    result["train_examples"] = len(train)
    result["edge_fraction"] = sum(ex["targets"]["edge"].decoded for ex in train) / len(train)
    report("edge training records (positive fraction)",
           f"{len(train)} ({result['edge_fraction']:.1%})")

    staged = staged_scaffold(r, R, name)
    sig = edge_signals()
    enum = enumerate_reference(staged, train, sig, r, tolerance=tol)
    if enum["solved"]:
        enum["held_out_max_error"] = exact_error(staged, held, sig, r, selections=enum["selections"])
        enum["held_out_accuracy"] = accuracy(staged, held, sig, r, selections=enum["selections"])
    else:
        enum["held_out_max_error"] = None
        enum["held_out_accuracy"] = None
        best = max(((accuracy(staged, train, sig, r, selections={**{n.name: 0 for n in staged.nodes},
                                                                 "shifted": a, "edge": b}), a, b)
                    for a in range(3) for b in range(16)))
        enum["best_train_accuracy"] = best[0]
        enum["best_selection"] = {"shifted": best[1], "edge": best[2]}
        enum["best_held_out_accuracy"] = accuracy(
            staged, held, sig, r,
            selections={**{n.name: 0 for n in staged.nodes}, "shifted": best[1], "edge": best[2]})
    ident = all_conforming(staged, train, sig, r, tolerance=tol)
    result["staged"] = {"enumerate_fit": enum, "space_size": space_size(staged),
                        "conforming": ident["count"], "unique": ident["unique"],
                        "selection": enum["selections"]}
    report("staged space/solved/exhausted/unique/seconds",
           f"{space_size(staged)} / {enum['solved']} / {enum['exhausted']} / "
           f"{enum['unique']} / {enum['seconds']:.3f}")
    report("staged held-out max error / accuracy",
           f"{enum['held_out_max_error']} / {enum['held_out_accuracy']}")
    dump(args.tag, result)

    grad_staged = gradient_arm(lambda: staged_scaffold(r, R, name),
                               tuple(range(args.gradient_seeds)), train, sig, r,
                               steps=400, lr=.15, init_noise=.5, label="grad staged",
                               tolerance=tol, held=held)
    result["staged"]["gradient"] = {"rows": grad_staged, "summary": summarise(grad_staged),
                                    "init_noise": .5}
    dump(args.tag, result)
    report("staged gradient successes",
           f"{result['staged']['gradient']['summary']['successes']}/{len(grad_staged)}")

    # ---- the flat control: the same task, undecomposed ----
    flat = flat_scaffold(r, R)
    result["flat"] = {"space_size": space_size(flat)}
    report("flat space", space_size(flat))
    t0 = time.perf_counter()
    partial = enumerate_reference(flat, train, sig, r, tolerance=tol,
                                  max_programs=args.flat_budget)
    result["flat"]["enumerate_partial"] = partial
    dump(args.tag, result)
    rate = partial["evaluated"] / max(1e-9, partial["seconds"])
    result["flat"]["programs_per_second"] = rate
    result["flat"]["projected_exhaustive_seconds"] = space_size(flat) / max(1e-9, rate)
    report("flat enumeration evaluated/exhausted/seconds/rate",
           f"{partial['evaluated']} / {partial['exhausted']} / {partial['seconds']:.1f} / "
           f"{rate:.0f} programs/s")
    report("flat projected exhaustive wall clock (s)",
           f"{result['flat']['projected_exhaustive_seconds']:.3g}")
    grad_flat = gradient_arm(lambda: flat_scaffold(r, R), tuple(range(args.gradient_seeds)),
                             train, sig, r, steps=400, lr=.15, init_noise=.5,
                             label="grad flat", tolerance=tol, held=held)
    result["flat"]["gradient"] = {"rows": grad_flat, "summary": summarise(grad_flat),
                                  "init_noise": .5}
    report("flat gradient successes",
           f"{result['flat']['gradient']['summary']['successes']}/{len(grad_flat)}")
    result["flat"]["random_control"] = random_reference(flat, train, sig, r, draws=1000,
                                                        tolerance=tol)

    # ---- stage B: apply the composed window module at every position ----
    if enum["solved"]:
        window_module = staged.harden(enum["selections"])
        wname = r.register_module(window_module)
        BT = bytes_type(R)
        positions = window_positions(R)
        caller = positional_caller(r, BT, positions, [wname])
        LAB = product(IDX, BOOL)
        errors, seconds = [], []
        for s in range(200, 200 + args.apply):
            pixels, probes = episode(s, R, split="test", objects=args.objects)
            labels = edge_labels(probes, R)
            data = Value.of(BT, pixels)
            t0 = time.perf_counter()
            got = caller.run({"observation": data}, registry=r)[0]["mapped"]
            seconds.append(time.perf_counter() - t0)
            target = Value.of(setof(LAB, len(positions)),
                              tuple((p, labels[p]) for p in positions))
            errors.append(max((abs(a - b) for a, b in zip(got.flat(), target.flat())), default=0.))
        result["apply"] = {"episodes": args.apply, "positions": len(positions),
                           "max_error": max(errors),
                           "caller_nodes": len(caller.nodes),
                           "median_seconds": sorted(seconds)[len(seconds) // 2]}
        report("stage B window applied at every position: max error", max(errors))
        report("stage B caller nodes", len(caller.nodes))

    dump(args.tag, result)


if __name__ == "__main__":
    main()
