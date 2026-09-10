"""POST-HOC (not pre-registered): how often does each arm's first solution generalize?

The pre-registered cost is programs to the first program that conforms on the
12 training episodes. After the arms ran, the flat and schema-only arms' first
solutions turned out never to generalize in 400 samples. This asks, with many
more exact uniform samples of the first solvable shell, what fraction of each
arm's first-shell conformers also conform on all 36 held-out episodes, and turns
that into an estimate of the expected programs to the first *generalizing*
solution: cost / fraction (Wilson 95% bounds). Labelled post-hoc everywhere it
is quoted; it moves no pre-registered verdict.

    python posthoc.py --gap 0 --arms N "N'" "N''" P --samples 20000
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
import time
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np          # noqa: E402
import arms                 # noqa: E402
import cache                # noqa: E402
import engine               # noqa: E402
import family               # noqa: E402
import prior                # noqa: E402
from family import SLOTS    # noqa: E402

OUT = HERE / "out"


def run(gap, arm, n, seed):
    d = json.loads((OUT / f"arm_gap{gap}_{arm.replace(chr(39), 'p')}.json").read_text())
    if d["first_solvable_tier"] is None:
        return {"arm": arm, "gap": gap, "solution_exists": False}
    pool_name, pr = arms.arm_table(prior.load_sources())[arm]
    c = cache.load(gap)
    train = engine.Episodes(c["episodes"][:arms.N_TRAIN])
    allep = engine.Episodes(c["episodes"])
    pool = family.pools(pool_name, arms.WIDTH)
    T = engine.Tables(train, pool)
    T48 = engine.Tables(allep, pool)
    tiers = prior.tiers(prior.slot_scores(pr, pool, train, arms.WIDTH), pool) if pr else \
        [engine.Restriction(pool)]
    R = tiers[d["first_solvable_tier"]]
    t0 = time.perf_counter()
    counter = engine.Counter(T, R, collect=True)
    assert str(counter.total) == d["tiers"][d["first_solvable_tier"]]["K"]
    samples = counter.sample(n, random.Random(seed))
    rows = np.array([[s[k] for k in SLOTS] for s in samples], dtype=np.int64)
    _, hits = engine.evaluate_batch(T48, rows)
    train_ok = bool(hits[:, :arms.N_TRAIN].all())
    gen = hits[:, arms.N_TRAIN:].all(axis=1)
    k = int(gen.sum())
    lo, hi = arms.wilson(k, n)
    cost = Fraction(int(d["expected_programs"]["num"]), int(d["expected_programs"]["den"]))
    return {"arm": arm, "gap": gap, "tier": d["first_solvable_tier"], "samples": n,
            "train_conform_all": train_ok, "generalize_count": k, "generalize_fraction": k / n,
            "wilson95": [lo, hi], "held_out_accuracy_mean": float(hits[:, arms.N_TRAIN:].mean()),
            "cost_to_conforming_log10": math.log10(float(cost)),
            "cost_to_generalizing_estimate_log10": (math.log10(float(cost) / (k / n)) if k else None),
            "cost_to_generalizing_lower_bound_log10": math.log10(float(cost) / hi),
            "cost_to_generalizing_upper_bound_log10": (math.log10(float(cost) / lo) if lo > 0 else None),
            "seconds": time.perf_counter() - t0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=int, default=0)
    ap.add_argument("--arms", nargs="*", default=["N", "N'", "N''", "P"])
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = [run(a.gap, arm, a.samples, a.seed) for arm in a.arms
           if (OUT / f"arm_gap{a.gap}_{arm.replace(chr(39), 'p')}.json").exists()]
    for r in out:
        print(r, flush=True)
    (OUT / f"posthoc_gap{a.gap}.json").write_text(json.dumps(out, indent=1))
