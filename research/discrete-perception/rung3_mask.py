"""Rung 3, through the discrete/positional route: foreground from raw pixels.

Stage A (structure, discretely).  A per-position module scaffold
`(position, image bytes) -> (position, foreground)` is searched exhaustively by
`tcn.search.enumerate_fit` over supervised (episode, position) records.  The
supervision is the generator's `object_ids` probe reduced to `>= 0`, which is
the same equivalence class as `depth == 0` (`audit.py` verifies the two agree).

Stage B (application, positionally).  The exported module is frozen, registered,
and applied at *every* position of a held-out image by the three-node
`insert`/`pair`/`map` pattern, and compared against the whole dense probe.

Stage C.  `filter` + `count` over the same result set: an aggregate readout over
an image, which the ladder's operator audit says does not exist.  It does exist
once the image is a set of positional records.

Nothing here relaxes an input binding: the module's addresses are computed from
its own position argument by `add`, and every free choice is a constant or an
operator at a fixed input -- the case relaxation is measured to be good at, and
the case enumeration settles outright.
"""
from __future__ import annotations

import argparse
import json
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
from tcn.graph import Signal
from tcn.operators import Registry
from tcn.search import enumerate_fit, evaluate, space_size
from tcn.types import BOOL, Value, integer, product, setof

# The byte values a comparison node may choose between.  24/30/43 is the
# renderer's background; the rest are distractors, so the choice is a real
# search.  Widening this to all 256 values is measured separately (`--pool
# full`), where it becomes the point at which brute force stops.
POOL = (24, 30, 43, 99, 150)
COUNT_T = integer(32, signed=False)


def module_scaffold(registry, resolution, pool=POOL, searched=True, truths=range(16)):
    """`(position, image bytes) -> (position, foreground)`.

    Free choices: which byte value each channel is compared against, and the two
    Boolean combinators.  |space| = |pool|^3 * |truths|^2.
    """
    REC = record_type(resolution)
    consts = tuple((f"byte_{v}", Value.of(BYTE, v)) for v in pool)
    consts += (("one", Value.of(IDX, 1)), ("two", Value.of(IDX, 2)))
    b = Builder(registry, (("rec", REC),), consts)
    b.add("pos", "project", ["rec"], params={"index": 0})
    b.add("obs", "project", ["rec"], params={"index": 1})
    b.add("g_at", "add", ["pos", "one"])
    b.add("b_at", "add", ["pos", "two"])
    b.add("red", "index", ["obs", "pos"])
    b.add("green", "index", ["obs", "g_at"])
    b.add("blue", "index", ["obs", "b_at"])
    for name, source, truth in (("cmp_r", "red", BACKGROUND[0]),
                                ("cmp_g", "green", BACKGROUND[1]),
                                ("cmp_b", "blue", BACKGROUND[2])):
        values = pool if searched else (truth,)
        b.choice(name, [("eq", (source, f"byte_{v}"), None) for v in values])
    b.choice("rg", [(f"truth_{t}", ("cmp_r", "cmp_g"), None) for t in (truths if searched else (8,))])
    b.choice("foreground", [(f"truth_{t}", ("rg", "cmp_b"), None) for t in (truths if searched else (7,))])
    b.add("record", "tuple", ["pos", "foreground"])
    return b.program((("y", "record"),))


def signals():
    return (Signal("foreground", "foreground", ("core",), BOOL, "bce"),)


def pixel_examples(seeds, resolution, split="train", per_image=None, seed=0, objects=6):
    """One supervised example per (episode, position): the dense probe, unaggregated.

    `per_image` subsamples positions -- that is the supervision budget, and the
    module is still applied at every position in stage B.
    """
    out = []
    starts = pixel_starts(resolution)
    REC = record_type(resolution)
    rng = random.Random(seed)
    for s in seeds:
        pixels, probes = episode(s, resolution, split=split, objects=objects)
        raw = Value.of(bytes_type(resolution), pixels).raw       # encode the image once
        chosen = range(len(starts)) if per_image is None else sorted(
            rng.sample(range(len(starts)), min(per_image, len(starts))))
        for i in chosen:
            rec = Value(REC, (starts[i], raw))                   # shares the encoded tuple
            out.append({"inputs": {"rec": rec},
                        "targets": {"foreground": Value.of(BOOL, probes["object_ids"][i] >= 0)}})
    rng.shuffle(out)
    return out


def keep_flag_module(registry, labelled):
    b = Builder(registry, (("rec", labelled),))
    b.add("flag", "project", ["rec"], params={"index": 1})
    return b.program((("y", "flag"),))


def target_set(labels, resolution):
    starts = pixel_starts(resolution)
    LAB = product(IDX, BOOL)
    return Value.of(setof(LAB, len(starts)), tuple((starts[i], labels[i]) for i in range(len(starts))))


def run(resolution, train_seeds, held_seeds, per_image, pool, gradient_seeds,
        apply_seeds, objects=6, tolerance=1e-6):
    r = Registry()
    result = {"resolution": resolution, "objects": objects,
              "observation_width": 3 * resolution * resolution,
              "positions": resolution * resolution,
              "train_seeds": list(train_seeds), "held_seeds": list(held_seeds),
              "per_image": per_image, "pool": list(pool)}

    t0 = time.perf_counter()
    train = pixel_examples(train_seeds, resolution, "train", per_image, seed=1, objects=objects)
    held = pixel_examples(held_seeds, resolution, "test", per_image, seed=2, objects=objects)
    result["data_seconds"] = time.perf_counter() - t0
    result["train_examples"] = len(train)
    result["train_foreground_fraction"] = sum(
        ex["targets"]["foreground"].decoded for ex in train) / len(train)
    report("training records (foreground fraction)",
           f"{len(train)} ({result['train_foreground_fraction']:.1%})")

    scaffold = module_scaffold(r, resolution, pool)
    sig = signals()
    result["space_size"] = space_size(scaffold)
    report("stage A discrete space", result["space_size"])

    # Time to the *first* conforming program, which forfeits the certificate.
    t0 = time.perf_counter()
    first = enumerate_fit(scaffold, train, sig, registry=r, tolerance=tolerance,
                          stop_at_first=True).to_dict()
    first["wall"] = time.perf_counter() - t0
    result["enumerate_first"] = first
    report("enumerate_fit stop_at_first solved/evaluated/seconds",
           f"{first['solved']} / {first['evaluated']} / {first['seconds']:.2f}")

    # The certified reference: exhaustive, exact, with a uniqueness verdict.
    enum = enumerate_reference(scaffold, train, sig, r, tolerance=tolerance)
    result["enumerate_fit"] = enum
    report("enumerate_fit solved/evaluated/exhausted/unique/seconds",
           f"{enum['solved']} / {enum['evaluated']} / {enum['exhausted']} / "
           f"{enum['unique']} / {enum['seconds']:.1f}")

    result["enumerate_fit"]["held_out_max_error"] = exact_error(
        scaffold, held, sig, r, selections=enum["selections"])
    result["enumerate_fit"]["held_out_accuracy"] = accuracy(
        scaffold, held, sig, r, selections=enum["selections"])
    report("held-out error of the program enumerate_fit returns",
           result["enumerate_fit"]["held_out_max_error"])

    # How identified is the program by this supervision, as supervision grows?
    # Conformance is monotone in the training set, so the sweep is run once at
    # the smallest budget and the survivors are filtered upward -- the whole
    # curve costs about one exhaustive sweep.
    curve, survivors = [], None
    budgets, b = [], 16
    while b < len(train):
        budgets.append(b); b *= 4
    budgets.append(len(train))
    budgets = sorted(set(budgets))
    seen = 0
    for b in budgets:
        t0 = time.perf_counter()
        if survivors is None:
            # Same sweep, same conforming set, with the fixed prefix computed once
            # per record instead of once per candidate program (see incremental.py,
            # which checks the two agree exactly).
            from incremental import incremental_conforming
            ident = incremental_conforming(scaffold, train[:b], sig, r, tolerance=tolerance)
            survivors = ident["conforming"]
            swept = ident["space_size"]
        else:
            survivors = [s for s in survivors
                         if (lambda e: e is not None and e <= tolerance)(
                             evaluate(scaffold, s, train[seen:b], sig, r, tolerance))]
            swept = 0
        seen = b
        generalising = [s for s in survivors
                        if (lambda e: e is not None and e <= tolerance)(
                            evaluate(scaffold, s, held, sig, r, tolerance))]
        curve.append({"supervised_records": b, "conforming": len(survivors),
                      "also_exact_on_held_out": len(generalising),
                      "unique": len(survivors) == 1,
                      "programs_swept": swept, "seconds": time.perf_counter() - t0})
        report(f"identification at {b:5d} records: conforming / generalising",
               f"{len(survivors)} / {len(generalising)} of {result['space_size']}")
    result["identification"] = {"space_size": result["space_size"], "curve": curve,
                                "conforming_train": curve[-1]["conforming"],
                                "conforming_held_out": curve[-1]["also_exact_on_held_out"],
                                "unique_on_train": curve[-1]["unique"]}
    result["conforming_examples"] = survivors[:8]

    rnd = random_reference(scaffold, train, sig, r, draws=2000, tolerance=tolerance)
    result["random_control"] = rnd
    report("random-draw solution density", f"{rnd['density']:.4g}")

    # Gradient path over the identical space and data.
    grad = gradient_arm(lambda: module_scaffold(r, resolution, pool), gradient_seeds,
                        train, sig, r, steps=400, lr=.15, init_noise=.5,
                        label=f"grad R{resolution}", tolerance=tolerance, held=held)
    result["gradient"] = {"init_noise": .5, "steps": 400, "lr": .15, "rows": grad,
                          "summary": summarise(grad)}
    report("gradient (init_noise 0.5) successes",
           f"{result['gradient']['summary']['successes']}/{len(grad)} "
           f"median {result['gradient']['summary']['median_seconds']:.1f}s")

    if not enum["solved"]:
        return result

    # Stage B: freeze the searched module and apply it at every position.
    learned = scaffold.harden(enum["selections"])
    reference = module_scaffold(r, resolution, pool, searched=False)

    def chosen(p):
        return [(n.name, n.candidates[n.selected].operator.to_dict()["parameters"],
                 n.candidates[n.selected].operator.name, n.candidates[n.selected].sources)
                for n in p.nodes]

    result["module_matches_renderer_test"] = chosen(learned) == chosen(reference)
    result["module"] = {"nodes": len(learned.nodes), "digest": learned.digest,
                        "execution_cost": learned.execution_cost(r),
                        "selection": enum["selections"]}
    report("enumerated module identical to the renderer's own test",
           result["module_matches_renderer_test"])

    name = r.register_module(learned)
    BT = bytes_type(resolution)
    starts = pixel_starts(resolution)
    caller = positional_caller(r, BT, starts, [name])
    result["caller_nodes"] = len(caller.nodes)
    report("stage B caller nodes for %d positions" % len(starts), len(caller.nodes))

    LAB = product(IDX, BOOL)
    keep = r.register_module(keep_flag_module(r, LAB))
    b = Builder(r, (("observation", BT),),
                (("positions", Value.of(setof(IDX, len(starts)), starts)),
                 ("empty", Value.of(setof(BT, 1), ()))))
    b.add("held", "insert", ["empty", "observation"])
    b.add("records", "pair", ["positions", "held"])
    b.add("mapped", "map", ["records"], params={"module": name})
    b.add("kept", "filter", ["mapped"], params={"module": keep})
    b.add("total", "count", ["kept"])
    counter = b.program((("y", "total"),))

    errors, count_errors, apply_seconds, fg = [], [], [], 0
    for s in apply_seeds:
        pixels, probes = episode(s, resolution, split="test", objects=objects)
        labels = [probes["object_ids"][i] >= 0 for i in range(len(starts))]
        fg += sum(labels)
        data = Value.of(BT, pixels)
        t0 = time.perf_counter()
        got = caller.run({"observation": data}, registry=r)[0]["mapped"]
        apply_seconds.append(time.perf_counter() - t0)
        target = target_set(labels, resolution)
        errors.append(max((abs(a - b) for a, b in zip(got.flat(), target.flat())), default=0.))
        total = counter.run({"observation": data}, registry=r)[0]["y"]
        count_errors.append(abs(total.decoded - sum(labels)))
    result["apply"] = {"episodes": len(apply_seeds), "positions": len(starts),
                       "pixels": len(apply_seeds) * len(starts),
                       "foreground_fraction": fg / max(1, len(apply_seeds) * len(starts)),
                       "max_error_over_dense_probe": max(errors),
                       "max_count_error": max(count_errors),
                       "median_seconds_per_image": sorted(apply_seconds)[len(apply_seconds) // 2]}
    report("stage B held-out max error over the whole dense probe", max(errors))
    report("stage C held-out max foreground-count error", max(count_errors))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolutions", type=int, nargs="+", default=[4, 8, 16, 24, 32])
    ap.add_argument("--train", type=int, default=6)
    ap.add_argument("--held", type=int, default=6)
    ap.add_argument("--per-image", type=int, default=48)
    ap.add_argument("--apply", type=int, default=3)
    ap.add_argument("--gradient-seeds", type=int, default=8)
    ap.add_argument("--objects", type=int, default=6)
    ap.add_argument("--tag", default="rung3_mask")
    args = ap.parse_args()

    rows = []
    for R in args.resolutions:
        print(f"\n=== resolution {R} ===", flush=True)
        rows.append(run(R, tuple(range(args.train)), tuple(range(100, 100 + args.held)),
                        args.per_image, POOL, tuple(range(args.gradient_seeds)),
                        tuple(range(200, 200 + args.apply)), objects=args.objects))
        dump(args.tag, {"rows": rows, "arguments": vars(args)})


if __name__ == "__main__":
    main()
