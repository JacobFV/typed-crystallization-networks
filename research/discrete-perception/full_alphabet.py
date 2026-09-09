"""Where brute force stops, and what to do instead -- measured, not asserted.

Rung 3 is solved by exhaustive enumeration when the byte constants come from a
five-value pool.  Widen that pool to the **whole 256-value byte alphabet** and
the same scaffold is 256^3 * 16^2 = 4,294,967,296 programs.  That is past
`enumerate_fit`, and the track's fourth question is what replaces it.

Four arms over the identical scaffold, data and objective:

* **brute force** with a budget: the measured evaluation rate and the projected
  exhaustive wall clock.
* **observed-alphabet scoping**: restrict each comparison's constants to the
  byte values that actually occur at that channel in the training observations.
  This uses only the observations, no probe and no renderer knowledge, and the
  space it leaves is measured rather than assumed.
* **coordinate-descent hill climbing with random restarts** over the same
  space, using `Program.execute` and the same conformance test -- a genuine
  search algorithm rather than an exhaustive one.
* **gradient descent** on the same scaffold, with explicit initialisation noise.

The hill climb's objective is training accuracy on a fixed subsample; a program
that reaches 1.0 there is then checked for exact conformance on the whole
training set, so a reported success is the same success `enumerate_fit` reports.
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

from common import (accuracy, dump, enumerate_reference, episode, exact_error, gradient_arm,
                    report, summarise)
from rung3_mask import module_scaffold, pixel_examples, signals
from tcn.operators import Registry
from tcn.search import enumerate_fit, evaluate, space_size

FULL = tuple(range(256))


def observed_alphabet(seeds, resolution, objects, split="train"):
    """Byte values that occur at each channel in the training observations."""
    channels = [set(), set(), set()]
    for s in seeds:
        pixels, _ = episode(s, resolution, split=split, objects=objects)
        for i in range(resolution * resolution):
            for c in range(3):
                channels[c].add(pixels[3 * i + c])
    return [tuple(sorted(x)) for x in channels]


def hill_climb(build, train, sig, registry, seeds, sample=32, passes=6, tolerance=1e-6,
               rng_seed=0):
    """Coordinate descent over node choices, restarted.

    At each step one node is chosen and every one of its candidates is scored
    with the others held fixed; the best is kept.  Ties are broken at random.
    A pass that improves nothing ends the restart.
    """
    program = build()
    names = [n.name for n in program.nodes]
    counts = [len(n.candidates) for n in program.nodes]
    free = [i for i, c in enumerate(counts) if c > 1]
    rows = []
    for s in seeds:
        rng = random.Random(rng_seed + s)
        subset = rng.sample(train, min(sample, len(train)))
        sel = {names[i]: rng.randrange(counts[i]) for i in range(len(names))}
        evaluated = 0
        t0 = time.perf_counter()
        best = accuracy(program, subset, sig, registry, selections=sel)
        evaluated += 1
        solved = False
        for _ in range(passes):
            improved = False
            for i in rng.sample(free, len(free)):
                name = names[i]
                scores = []
                for k in range(counts[i]):
                    trial = dict(sel); trial[name] = k
                    scores.append((accuracy(program, subset, sig, registry, selections=trial), k))
                    evaluated += 1
                top = max(scores)[0]
                pick = rng.choice([k for a, k in scores if a == top])
                if top > best or (top == best and pick != sel[name]):
                    improved |= top > best
                    best, sel[name] = top, pick
            if best >= 1. - 1e-12:
                err = evaluate(program, sel, train, sig, registry, tolerance)
                if err is not None and err <= tolerance:
                    solved = True
                    break
            if not improved:
                break
        rows.append({"restart": s, "solved": solved, "subset_accuracy": best,
                     "evaluations": evaluated, "seconds": time.perf_counter() - t0,
                     "selections": sel if solved else None})
        report(f"  hill climb restart {s}: solved={solved} evals={evaluated} "
               f"acc={best:.4f}", f"{rows[-1]['seconds']:.1f}s")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=8)
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--held", type=int, default=6)
    ap.add_argument("--per-image", type=int, default=48)
    ap.add_argument("--objects", type=int, default=6)
    ap.add_argument("--brute-budget", type=int, default=100000)
    ap.add_argument("--restarts", type=int, default=6)
    ap.add_argument("--gradient-seeds", type=int, default=6)
    ap.add_argument("--tag", default="full_alphabet")
    args = ap.parse_args()

    R = args.resolution
    train_seeds = tuple(range(args.train))
    held_seeds = tuple(range(100, 100 + args.held))
    tol = 1e-6
    r = Registry()
    train = pixel_examples(train_seeds, R, "train", args.per_image, seed=1, objects=args.objects)
    held = pixel_examples(held_seeds, R, "test", args.per_image, seed=2, objects=args.objects)
    sig = signals()
    result = {"arguments": vars(args), "train_examples": len(train)}

    full = module_scaffold(r, R, FULL)
    result["space_size"] = space_size(full)
    report("full-alphabet space", result["space_size"])

    # 1. brute force with a budget
    brute = enumerate_reference(full, train, sig, r, tolerance=tol,
                                max_programs=args.brute_budget)
    brute["programs_per_second"] = brute["evaluated"] / max(1e-9, brute["seconds"])
    brute["projected_exhaustive_seconds"] = (result["space_size"] /
                                             max(1e-9, brute["programs_per_second"]))
    result["brute_force"] = brute
    report("brute force evaluated/solved/exhausted/rate",
           f"{brute['evaluated']} / {brute['solved']} / {brute['exhausted']} / "
           f"{brute['programs_per_second']:.0f} programs/s")
    report("brute force projected exhaustive wall clock (s)",
           f"{brute['projected_exhaustive_seconds']:.4g}")

    # 2. observed-alphabet scoping
    alphabet = observed_alphabet(train_seeds, R, args.objects)
    result["observed_alphabet"] = {"sizes": [len(a) for a in alphabet]}
    scoped_pool = tuple(sorted(set(alphabet[0]) | set(alphabet[1]) | set(alphabet[2])))
    scoped = module_scaffold(r, R, scoped_pool)
    result["observed_alphabet"]["pool_size"] = len(scoped_pool)
    result["observed_alphabet"]["space_size"] = space_size(scoped)
    result["observed_alphabet"]["reduction"] = result["space_size"] / space_size(scoped)
    report("observed-alphabet pool / space / reduction",
           f"{len(scoped_pool)} / {space_size(scoped)} / "
           f"{result['observed_alphabet']['reduction']:.1f}x")
    scoped_enum = enumerate_reference(scoped, train, sig, r, tolerance=tol,
                                      max_programs=args.brute_budget)
    scoped_enum["programs_per_second"] = scoped_enum["evaluated"] / max(1e-9, scoped_enum["seconds"])
    scoped_enum["projected_exhaustive_seconds"] = (space_size(scoped) /
                                                   max(1e-9, scoped_enum["programs_per_second"]))
    result["observed_alphabet"]["enumeration"] = scoped_enum
    report("scoped enumeration evaluated/solved/exhausted/projected s",
           f"{scoped_enum['evaluated']} / {scoped_enum['solved']} / {scoped_enum['exhausted']} / "
           f"{scoped_enum['projected_exhaustive_seconds']:.4g}")

    # 3. hill climbing over the full space
    rows = hill_climb(lambda: module_scaffold(r, R, FULL), train, sig, r,
                      range(args.restarts), sample=32, passes=6, tolerance=tol)
    solved = [x for x in rows if x["solved"]]
    result["hill_climb"] = {"restarts": rows, "successes": len(solved),
                            "median_evaluations": sorted(x["evaluations"] for x in rows)[len(rows) // 2],
                            "median_seconds": sorted(x["seconds"] for x in rows)[len(rows) // 2]}
    if solved:
        result["hill_climb"]["held_out_max_error"] = exact_error(
            full, held, sig, r, selections=solved[0]["selections"])
    report("hill climb successes / median evaluations",
           f"{len(solved)}/{len(rows)} / {result['hill_climb']['median_evaluations']}")

    # 4. gradient descent on the same space
    grad = gradient_arm(lambda: module_scaffold(r, R, FULL), tuple(range(args.gradient_seeds)),
                        train, sig, r, steps=400, lr=.15, init_noise=.5,
                        label="grad full", tolerance=tol, held=held)
    result["gradient"] = {"rows": grad, "summary": summarise(grad), "init_noise": .5}
    report("gradient successes",
           f"{result['gradient']['summary']['successes']}/{len(grad)}")

    dump(args.tag, result)


if __name__ == "__main__":
    main()
