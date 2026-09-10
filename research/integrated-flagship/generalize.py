"""Exact counts of GENERALIZING programs: conform on all 48 episodes (12 + 36).

Settles two questions with a certificate rather than a sample:

1. Does the matched distractor space contain ANY program that conforms on the
   training and held-out episodes together? If not, D'' could not succeed by
   construction (a straw-man distractor), and its 0/400 is uninformative.
2. For the schema pool (N', and N''s tiers), the exact number of generalizing
   programs, hence the exact expected programs to the first generalizing one
   under each arm's order (POST-HOC: not the pre-registered metric).

The 48-episode count uses the counter's direct mode (no 2^48 transform); direct
mode is checked equal to the transform on the 12 training episodes first.
Plus an analytic fact: which distractor address expressions equal length-5.

    python generalize.py --gap 0
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
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

OUT = HERE / "out"


def count(T, R, direct):
    c = engine.Counter.__new__(engine.Counter)
    c.direct = direct
    engine.Counter.__init__(c, T, R)
    return c.total


def tiered_cost(T, tiers, Ttrain):
    before, prev_S, prev_K, rows = 0, 0, 0, []
    cost = None
    for t, R in enumerate(tiers):
        R.occ_mask = (1 << arms.N_TRAIN) - 1
        # S is the arm's own (training) space; K counts its programs that conform on all 48
        S, K = engine.space_size(Ttrain, R), count(T, R, True)
        rows.append({"tier": t, "S": str(S), "K48": str(K)})
        if cost is None and K - prev_K > 0 and prev_K == 0:
            cost = Fraction(before) + Fraction(S - prev_S + 1, K - prev_K + 1)
        elif cost is None:
            before += S - prev_S
        prev_S, prev_K = S, K
        if cost is not None:
            break
    return cost, rows


def main(gap):
    d = cache.load(gap)
    train = engine.Episodes(d["episodes"][:arms.N_TRAIN])
    allep = engine.Episodes(d["episodes"])
    out = {"gap": gap, "episodes_all": allep.E}
    # direct mode == transform, on the training episodes
    agree = {}
    for name in ("distractor", "schema"):
        pool = family.pools(name, arms.WIDTH)
        T = engine.Tables(train, pool)
        R = engine.Restriction(pool)
        agree[name] = {"transform": str(count(T, R, False)), "direct": str(count(T, R, True))}
    out["direct_equals_transform_on_train"] = agree
    # analytic: which distractor address expressions equal length-5 on all 48 episodes
    pool = family.pools("distractor", arms.WIDTH)
    want = allep.length - 5
    hits = [str(s) for s in pool["cpos"]
            if (engine.addr_values(s, allep.length) == want).all()]
    out["distractor_addresses_equal_length_minus_5"] = hits
    out["distractor_address_pool_size"] = len(pool["cpos"])
    # exact generalizing counts
    for name in ("distractor", "schema"):
        pool = family.pools(name, arms.WIDTH)
        T48 = engine.Tables(allep, pool)
        t0 = time.perf_counter()
        R = engine.Restriction(pool)
        K48 = count(T48, R, True)
        S = engine.space_size(T48, R)
        out[name] = {"S_all_episodes": str(S), "K_generalizing": str(K48),
                     "certificate": "unique" if K48 == 1 else "complete",
                     "expected_programs_to_first_generalizing_uniform":
                         {"log10": math.log10(float(Fraction(S + 1, K48 + 1)))} if K48 else None,
                     "seconds": time.perf_counter() - t0}
        print(name, out[name], flush=True)
    # N'': its tiers, counting generalizing programs (order fixed by the training-fitted prior)
    src = prior.load_sources()
    pool = family.pools("schema", arms.WIDTH)
    scores = prior.slot_scores(prior.Prior(src), pool, train, arms.WIDTH)
    tiers = prior.tiers(scores, pool)
    T48 = engine.Tables(allep, pool)
    cost, rows = tiered_cost(T48, tiers, engine.Tables(train, pool))
    out["N''_tiers_generalizing"] = rows
    out["N''_expected_programs_to_first_generalizing"] = (
        {"log10": math.log10(float(cost)), "num": str(cost.numerator), "den": str(cost.denominator)}
        if cost else None)
    (OUT / f"generalize_gap{gap}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:3000])


SCHEMA_ARMS = ["N'", "N''", "A_noobs"] + [f"R_{s}" for s in range(5)] + \
    ["U_feat", "U_steep", "U_Vonly", "KO_ADDR", "KO_TRUTH", "KO_STEP", "STEP_only", "STEP_hand"] + \
    [f"OCC_{r}" for r in (1, 3, 10, 100, 3500)] + [f"V_{r}" for r in (1, 3, 10, 100)]


def first_generalizing(gap, arm, table, train, T48, Ttrain):
    """POST-HOC: exact expected programs to the first program conforming on all
    48 episodes, visiting the arm's own tiers (uniform order inside a shell)."""
    pool_name, pr = table[arm]
    pool = family.pools(pool_name, arms.WIDTH)
    tiers = prior.tiers(prior.slot_scores(pr, pool, train, arms.WIDTH), pool) if pr else \
        [engine.Restriction(pool)]
    before, prev_S, prev_K, rows, cost = 0, 0, 0, [], None
    t0 = time.perf_counter()
    for t, R in enumerate(tiers):
        R.occ_mask = (1 << arms.N_TRAIN) - 1   # 'occurs' decided on training, as in the arm
        S = engine.space_size(Ttrain, R)      # the arm's own (training) space
        K = count(T48, R, True)
        rows.append({"tier": t, "S": str(S), "K48": str(K)})
        if K - prev_K > 0 and prev_K == 0:
            cost = Fraction(before) + Fraction(S - prev_S + 1, K - prev_K + 1)
            break
        before += S - prev_S
        prev_S, prev_K = S, K
    return {"arm": arm, "gap": gap, "tiers": rows,
            "first_generalizing_tier": len(rows) - 1 if cost is not None else None,
            "expected_programs_to_first_generalizing":
                ({"num": str(cost.numerator), "den": str(cost.denominator),
                  "log10": math.log10(cost.numerator) - math.log10(cost.denominator)}
                 if cost is not None else None),
            "certificate": "complete", "seconds": time.perf_counter() - t0}


def main_all(gap, which):
    d = cache.load(gap)
    train = engine.Episodes(d["episodes"][:arms.N_TRAIN])
    allep = engine.Episodes(d["episodes"])
    table = arms.arm_table(prior.load_sources())
    T48 = engine.Tables(allep, family.pools("schema", arms.WIDTH))
    Ttrain = engine.Tables(train, family.pools("schema", arms.WIDTH))
    out = []
    for arm in which:
        if not (OUT / f"arm_gap{gap}_{arm.replace(chr(39), 'p')}.json").exists():
            continue
        r = first_generalizing(gap, arm, table, train, T48, Ttrain)
        out.append(r)
        print({k: v for k, v in r.items() if k != "tiers"}, flush=True)
        (OUT / f"first_generalizing_gap{gap}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.all:
        main_all(a.gap, SCHEMA_ARMS)
    else:
        main(a.gap)
