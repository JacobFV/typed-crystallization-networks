"""Non-uniqueness is a statement about the probe -- so what should pick the program?

`enumerate_fit` returns the first conforming program in candidate order.  When
the solution is not unique that is a lexicographic tie-break, and a tie-break is
a prior.  Measured here on rung 3, over three disjoint episode ranges:

  train      -> the conforming set
  validation -> a filter on that set (still supervision, just more of it)
  test       -> the delivered module applied at every position by the three-node
                caller, scored against the whole dense probe

Three selection rules are compared on the same conforming set: lexicographic
(what `enumerate_fit` does today), validation-filtered lexicographic, and the
program gradient descent converges to.  A rule that is right on test only because
it was chosen on test would be cheating; test episodes are never used to select.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import (BYTE, IDX, Builder, accuracy, all_conforming, bytes_type, dump, episode,
                    exact_error, local_fit, pixel_starts, positional_caller, report)
from rung3_mask import POOL, module_scaffold, pixel_examples, signals, target_set
from tcn.operators import Registry
from tcn.search import evaluate, space_size
from tcn.types import BOOL, Value, product, setof


def apply_everywhere(registry, resolution, selections, scaffold, seeds, objects):
    """Freeze the module and run it at every position of fresh episodes."""
    module = scaffold.harden(selections)
    name = registry.register_module(module)
    BT = bytes_type(resolution)
    starts = pixel_starts(resolution)
    caller = positional_caller(registry, BT, starts, [name])
    LAB = product(IDX, BOOL)
    worst, wrong, total = 0., 0, 0
    for s in seeds:
        pixels, probes = episode(s, resolution, split="test", objects=objects)
        labels = [probes["object_ids"][i] >= 0 for i in range(len(starts))]
        got = caller.run({"observation": Value.of(BT, pixels)}, registry=registry)[0]["mapped"]
        target = target_set(labels, resolution)
        a, b = got.flat(), target.flat()
        worst = max(worst, max((abs(x - y) for x, y in zip(a, b)), default=0.))
        wrong += sum(abs(x - y) > 1e-6 for x, y in zip(a, b))
        total += len(a)
    return {"max_error": worst, "wrong_slots": wrong, "slots": total,
            "accuracy": 1 - wrong / max(1, total), "caller_nodes": len(caller.nodes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolutions", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--validation", type=int, default=8)
    ap.add_argument("--test", type=int, default=8)
    ap.add_argument("--objects", type=int, default=6)
    ap.add_argument("--gradient-seeds", type=int, default=6)
    ap.add_argument("--tag", default="tiebreak")
    args = ap.parse_args()

    tol = 1e-6
    rows = []
    for R in args.resolutions:
        r = Registry()
        train = pixel_examples(tuple(range(args.train)), R, "train", None, seed=1,
                               objects=args.objects)
        valid = pixel_examples(tuple(range(100, 100 + args.validation)), R, "test", None,
                               seed=2, objects=args.objects)
        scaffold = module_scaffold(r, R, POOL)
        sig = signals()
        t0 = time.perf_counter()
        ident = all_conforming(scaffold, train, sig, r, tolerance=tol)
        conf = ident["conforming"]
        filtered = [s for s in conf
                    if (lambda e: e is not None and e <= tol)(
                        evaluate(scaffold, s, valid, sig, r, tol))]
        row = {"resolution": R, "space_size": space_size(scaffold),
               "train_records": len(train), "validation_records": len(valid),
               "conforming_train": len(conf), "conforming_train_and_validation": len(filtered),
               "sweep_seconds": time.perf_counter() - t0}
        report(f"R={R} conforming on train / + validation / of space",
               f"{len(conf)} / {len(filtered)} / {row['space_size']}")

        test_seeds = tuple(range(200, 200 + args.test))
        picks = {"lexicographic": conf[0] if conf else None,
                 "validation_filtered": filtered[0] if filtered else None}
        grad_rows = []
        for seed in range(args.gradient_seeds):
            model, info = local_fit(module_scaffold(r, R, POOL), train, sig, steps=400,
                                    lr=.15, registry=r, tolerance=tol, init_noise=.5, seed=seed)
            grad_rows.append({"seed": seed, "ok": bool(info["exact_conformance"]),
                              "selections": info["selections"]})
        row["gradient_rows"] = grad_rows
        ok = [g for g in grad_rows if g["ok"]]
        picks["gradient"] = ok[0]["selections"] if ok else None
        row["gradient_successes"] = len(ok)
        row["gradient_distinct_programs"] = len({tuple(sorted(g["selections"].items()))
                                                 for g in ok})

        row["rules"] = {}
        for rule, sel in picks.items():
            if sel is None:
                row["rules"][rule] = None
                continue
            out = apply_everywhere(r, R, sel, scaffold, test_seeds, args.objects)
            out["selections"] = sel
            out["in_conforming_set"] = sel in conf
            row["rules"][rule] = out
            report(f"  R={R} {rule:20s} test accuracy / max error",
                   f"{out['accuracy']:.6f} / {out['max_error']}")

        # how often would each rule be wrong?  score every conforming program.
        scored = []
        for sel in (conf if len(conf) <= 4000 else conf[:4000]):
            scored.append(evaluate(scaffold, sel, valid, sig, r, tol))
        row["fraction_conforming_that_fail_validation"] = sum(
            1 for e in scored if e is None or e > tol) / max(1, len(scored))
        report(f"  R={R} fraction of conforming programs failing validation",
               f"{row['fraction_conforming_that_fail_validation']:.4f}")
        rows.append(row)
        dump(args.tag, {"rows": rows, "arguments": vars(args)})


if __name__ == "__main__":
    main()
