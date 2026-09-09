"""The same rung with the threshold pool widened to every value 0..511.

`residual.py` shows the coarse eight-value pool omits the whole separating
interval T in [140, 158], so the coarse search correctly returns the best
program available to it and that program is wrong on two held-out positions.
Widening the pool is the fix, and it is also the experiment the previous track's
section 8 sets up: a choice among many CONSTANTS at a fixed input is the case
relaxation was measured to be 208x better than chance at and enumeration was
measured to lose its certificate on.

The difference here is that this constant sits BEHIND `pack`, which declares
`gradient="none"`.  So this arm is the direct test of which half of the method
boundary dominates when the two conditions conflict.
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "discrete-perception"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import (accuracy, exact_error, gradient_arm, random_reference, report, summarise)
from incremental import incremental_conforming
from common2 import OUT
from rung4_segment import (collinear_scaffold, choice_gradients, dump, same_examples,
                           same_signals, select)
from tcn.operators import Registry
from tcn.search import enumerate_fit, space_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=8)
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--validation", type=int, default=8)
    ap.add_argument("--held", type=int, default=8)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--gradient-seeds", type=int, default=4)
    ap.add_argument("--enum-budget", type=int, default=40000)
    ap.add_argument("--objects", type=int, default=6)
    ap.add_argument("--tag", default="wide_threshold")
    args = ap.parse_args()

    R = args.resolution; tol = 1e-6; r = Registry()
    thresholds = tuple(range(args.width))
    train = same_examples(range(args.train), R, "train", None, seed=3, objects=args.objects)
    val = same_examples(range(50, 50 + args.validation), R, "train", None, seed=5,
                        objects=args.objects)
    held = same_examples(range(100, 100 + args.held), R, "test", None, seed=4,
                         objects=args.objects)
    sig = same_signals()
    build = lambda: collinear_scaffold(r, R, thresholds=thresholds)
    program = build()
    n = space_size(program)
    result = {"arguments": vars(args), "space_size": n, "threshold_pool": args.width,
              "train_examples": len(train), "validation_examples": len(val),
              "held_examples": len(held)}
    report("[wide] space", n)

    # reference cost of the shipped loop, capped, plus its projection
    t0 = time.perf_counter()
    capped = enumerate_fit(program, train, sig, registry=r, tolerance=tol,
                           max_programs=args.enum_budget)
    wall = time.perf_counter() - t0
    rate = capped.evaluated / max(1e-9, wall)
    result["enumerate_fit_capped"] = capped.to_dict() | {
        "wall": wall, "programs_per_second": rate,
        "projected_exhaustive_seconds": n / max(1e-9, rate)}
    report("[wide] enumerate_fit capped: evaluated/solved/rate/projected exhaustive s",
           f"{capped.evaluated}/{capped.solved}/{rate:.0f}/"
           f"{result['enumerate_fit_capped']['projected_exhaustive_seconds']:.4g}")

    t0 = time.perf_counter()
    first = enumerate_fit(program, train, sig, registry=r, tolerance=tol, stop_at_first=True)
    result["stop_at_first"] = first.to_dict() | {"wall": time.perf_counter() - t0}
    report("[wide] stop-at-first: evaluated / s",
           f"{first.evaluated} / {result['stop_at_first']['wall']:.2f}")

    inc = incremental_conforming(program, train, sig, r, tolerance=tol)
    conforming = inc["conforming"]
    result["incremental"] = {k: v for k, v in inc.items() if k != "conforming"}
    result["conforming_on_train"] = len(conforming)
    report("[wide] prefix-reusing exhaustive sweep: conforming / node evaluations / s",
           f"{len(conforming)} / {inc.get('node_evaluations', inc.get('evaluated'))} / "
           f"{inc['seconds']:.1f}")

    survivors = select(program, conforming, val, sig, r, tolerance=tol)
    result["conforming_and_validation_exact"] = len(survivors)
    report("[wide] also exact at every validation position", f"{len(survivors)} of {len(conforming)}")
    thr_values = sorted({thresholds[s["thr"]] for s in survivors})
    result["surviving_thresholds"] = thr_values
    report("[wide] thresholds among survivors", f"{thr_values[:12]}{'...' if len(thr_values)>12 else ''}")

    for label, sel in (("lexicographic", conforming[0] if conforming else None),
                       ("validation_filtered", survivors[0] if survivors else None)):
        if sel is None:
            result[label] = None; continue
        result[label] = {"selections": sel,
                         "threshold": thresholds[sel["thr"]],
                         "held_max_error": exact_error(program, held, sig, r, selections=sel),
                         "held_accuracy": accuracy(program, held, sig, r, selections=sel)}
        report(f"[wide] {label}: T / held max error / held accuracy",
               f"{result[label]['threshold']} / {result[label]['held_max_error']} / "
               f"{result[label]['held_accuracy']:.6f}")
    # how many of the conforming programs are actually right on fresh episodes
    exact_held = sum(1 for s in conforming
                     if exact_error(program, held, sig, r, selections=s) <= tol)
    result["conforming_and_held_exact"] = exact_held
    report("[wide] conforming programs that are also exact on the held-out split",
           f"{exact_held} of {len(conforming)}")

    result["random_density"] = random_reference(program, train, sig, r, 400, seed=1, tolerance=tol)
    result["choice_gradients"] = choice_gradients(build, train[:64], sig, r)
    report("[wide] choice logit |grad|_1", json.dumps(result["choice_gradients"]["choice_grad_l1"]))
    dump(args.tag, result)

    rows = gradient_arm(build, tuple(range(args.gradient_seeds)), train, sig, r,
                        steps=400, lr=.15, init_noise=.5, label="wide", tolerance=tol, held=held)
    result["gradient"] = {"rows": rows, "summary": summarise(rows), "init_noise": .5,
                          "held_exact": sum(1 for x in rows if x.get("held_err") == 0.)}
    report("[wide] gradient successes / held-out exact",
           f"{result['gradient']['summary']['successes']}/{len(rows)} / "
           f"{result['gradient']['held_exact']}")
    dump(args.tag, result)


if __name__ == "__main__":
    main()
