"""Plain argmax matched to a scheduler arm's *forward passes*, not its steps.

The step-matched `argmax@<arm>` control equalizes `optimizer.step()` calls, which
is track 1's standard. Perturbation scoring, though, buys its selection with
forward passes and no gradients, so on any budget where a scheduler arm claims a
win the honest follow-up is to hand plain argmax that arm's *forward* budget as
extra optimizer steps -- strictly generous to argmax, since each of those steps
also carries a backward pass the scheduler's sweep never paid for.

An argmax run with E extra steps performs `base + E` steps and `base + E + 1`
forward passes, so E is chosen per budget to reach the scheduler arm's mean
forward count.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from run_mixed import run  # noqa: E402


def one(seed, budget, extra, noise, label):
    record = run("argmax", seed, steps=budget, extra_steps=extra, noise=noise)
    record["arm"] = label
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--extra", type=int, required=True,
                    help="extra optimizer steps, chosen so total forwards match the target arm")
    ap.add_argument("--label", required=True)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, s, args.budget, args.extra, args.noise, args.label)
                   for s in range(args.seeds)]
        with out.open("w") as fh:
            for f in futures:
                fh.write(json.dumps(f.result()) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
