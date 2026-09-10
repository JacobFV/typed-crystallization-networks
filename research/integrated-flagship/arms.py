"""Every arm of PREREGISTRATION §4, exactly, from the recorded episodes.

    python arms.py --gap 0 --arm N        # one arm -> out/arm_gap0_N.json
    python arms.py --gap 0 --arm extras   # hard vector H + baselines -> out/extras_gap0.json

For each arm: the tiers its order visits (one tier for a uniform-order arm),
and per tier the exact space size S_t and conformer count K_t from
`engine.Counter`. The expected programs to the first conforming program is

    sum_{t < t*} (S_t - S_{t-1})  +  (S_shell + 1) / (K_shell + 1)

(uniform random order without replacement inside the first shell that holds a
conformer), kept as an exact fraction. Episodes = programs x 12 (every program
is rolled out on every training episode, as `enumerate_environment` does).

What the search finds first is a uniformly random conformer of that shell;
`--samples` of them are drawn exactly (`Counter.sample`) and scored on the 36
held-out episodes.
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

import numpy as np                    # noqa: E402
import cache                          # noqa: E402
import engine                         # noqa: E402
import family                         # noqa: E402
import prior                          # noqa: E402
from family import KIND, SLOTS        # noqa: E402

OUT = HERE / "out"
N_TRAIN = 12
WIDTH = 16
PERMUTATION_SEEDS = (0, 1, 2, 3, 4)


class _Rule:
    """A fixed, hand-written score: 1 when the named boolean feature is on, else `off`."""

    def __init__(self, feature=None, off=.5):
        self.feature, self.off = feature, off

    def score(self, row):
        if self.feature is None:
            return 1.0
        return 1.0 if row[self.feature] else self.off

    def describe(self):
        return {"rule": f"1 if {self.feature} else {self.off}" if self.feature else "uniform",
                "p0": None, "rows": 0, "features": [self.feature] if self.feature else [],
                "cells": {}}


class HandFeaturePrior:
    """POST-HOC control (coordinator-requested, not pre-registered): N''s features
    with UNFITTED weights -- 'prefer candidates whose feature is on' -- so the
    earlier domains' rows teach nothing here."""

    def __init__(self):
        self.est = {"ADDR": _Rule("V"), "LIT": _Rule("occurs"), "TRUTH": _Rule(), "STEP": _Rule()}

    def describe(self):
        return {k: e.describe() for k, e in self.est.items()}


class OccursRatioPrior:
    """POST-HOC sensitivity (coordinator-requested): N''s fitted prior, except the
    LIT estimator is pinned so that score(occurs) / score(absent) = `ratio`."""

    def __init__(self, src, ratio):
        base = prior.Prior(src)
        occ = base.est["LIT"].score({"occurs": True})
        self.est = dict(base.est)
        self.est["LIT"] = _Pinned(occ, occ / ratio, ratio)
        self._base = base

    def describe(self):
        d = self._base.describe()
        d["LIT"] = self.est["LIT"].describe()
        return d


class _Pinned:
    def __init__(self, on, off, ratio):
        self.on, self.off, self.ratio = on, off, ratio

    def score(self, row):
        return self.on if row["occurs"] else self.off

    def describe(self):
        return {"rule": f"occurs ratio pinned at {self.ratio}", "p0": None, "rows": 0,
                "features": ["occurs"],
                "cells": {"(False,)": {"survive": None, "n": None, "score": self.off},
                          "(True,)": {"survive": None, "n": None, "score": self.on}}}


OCC_RATIOS = (1, 3, 10, 100, 3500)
V_RATIOS = (1, 3, 10, 100)


class _VPinned:
    """N''s fitted ADDR estimator evaluated as if V were on, times 1/ratio when V is off:
    the op/kinds ordering is kept and only the V penalty is set."""

    def __init__(self, base, ratio):
        self.base, self.ratio = base, ratio

    def score(self, row):
        s = self.base.score(dict(row, V=True))
        return s if row["V"] else s / self.ratio

    def describe(self):
        d = self.base.describe()
        d["rule"] = f"V ratio pinned at {self.ratio} over the fitted V=True cells"
        return d


class VRatioPrior:
    """POST-HOC sensitivity (coordinator-requested): N''s fitted prior, ADDR's V
    penalty pinned at `ratio`, LIT left at N''s fitted values."""

    def __init__(self, src, ratio):
        base = prior.Prior(src)
        self.est = dict(base.est)
        self.est["ADDR"] = _VPinned(base.est["ADDR"], ratio)
        self._base = base

    def describe(self):
        d = self._base.describe()
        d["ADDR"] = self.est["ADDR"].describe()
        return d


class SteepHandPrior:
    """POST-HOC control (coordinator-requested): hand rules at N''s own steepness,
    NOTHING fitted from sources.json -- occurs 3511:1 (N''s fitted LIT ratio,
    hard-coded), V 10:1, TRUTH and STEP uniform."""

    def __init__(self):
        self.est = {"ADDR": _Rule("V", .1), "LIT": _Rule("occurs", 1 / 3511.),
                    "TRUTH": _Rule(), "STEP": _Rule()}

    def describe(self):
        return {k: e.describe() for k, e in self.est.items()}


class VOnlyHandPrior:
    """POST-HOC control (coordinator-requested): ONE hand rule, V 10:1 on the address
    slots, no op/kinds cells; LIT, TRUTH and STEP uniform; nothing fitted."""

    def __init__(self):
        self.est = {"ADDR": _Rule("V", .1), "LIT": _Rule(), "TRUTH": _Rule(), "STEP": _Rule()}

    def describe(self):
        return {k: e.describe() for k, e in self.est.items()}


class KnockoutPrior:
    """POST-HOC per-kind knockout (coordinator-requested): N''s fitted prior with
    exactly ONE learned estimator replaced -- by uniform for TRUTH/STEP, and by the
    hand V 10:1 rule for ADDR (so it is not A_noobs). A knockout that removes
    generalization names the domain whose rows carry it: ADDR <- s19/s23,
    TRUTH/STEP <- s33."""

    def __init__(self, src, kind):
        base = prior.Prior(src)
        self.est = dict(base.est)
        self.est[kind] = _Rule("V", .1) if kind == "ADDR" else _Rule()
        self.kind = kind

    def describe(self):
        return {k: e.describe() for k, e in self.est.items()}


class _ClassRule:
    """Hand STEP rule: 1 if the offset class is one of `good`, else `off`."""

    def __init__(self, good, off):
        self.good, self.off = set(good), off

    def score(self, row):
        return 1.0 if row["offset_class"] in self.good else self.off

    def describe(self):
        return {"rule": f"1 if offset_class in {sorted(self.good)} else {self.off}", "p0": None,
                "rows": 0, "features": ["offset_class"], "cells": {}}


class StepOnlyPrior:
    """POST-HOC sufficiency test (coordinator-requested): STEP fitted from
    sources.json (the 20 s33 rows); ADDR, LIT, TRUTH uniform; no V, no occurs."""

    def __init__(self, src):
        self.est = {"ADDR": _Rule(), "LIT": _Rule(), "TRUTH": _Rule(),
                    "STEP": prior.Prior(src).est["STEP"]}

    def describe(self):
        return {k: e.describe() for k, e in self.est.items()}


class StepHandPrior:
    """POST-HOC control (coordinator-requested): the one-line human STEP rule
    '1 if offset_class in {pixel, row} else 0.1', all else uniform, nothing fitted."""

    def __init__(self):
        self.est = {"ADDR": _Rule(), "LIT": _Rule(), "TRUTH": _Rule(),
                    "STEP": _ClassRule(("pixel", "row"), .1)}

    def describe(self):
        return {k: e.describe() for k, e in self.est.items()}


def arm_table(src):
    learned = prior.Prior(src)
    return {
        "STEP_only": ("schema", StepOnlyPrior(src)),
        "STEP_hand": ("schema", StepHandPrior()),
        **{f"KO_{k}": ("schema", KnockoutPrior(src, k)) for k in ("ADDR", "TRUTH", "STEP")},
        "U_Vonly": ("schema", VOnlyHandPrior()),
        "U_steep": ("schema", SteepHandPrior()),
        **{f"OCC_{r}": ("schema", OccursRatioPrior(src, r)) for r in OCC_RATIOS},
        **{f"V_{r}": ("schema", VRatioPrior(src, r)) for r in V_RATIOS},
        "U_feat": ("schema", HandFeaturePrior()),
        "N": ("flat", None), "N'": ("schema", None), "N''": ("schema", learned),
        "P": ("flat", learned), "D'": ("distractor", None),
        **{f"D''_{s}": ("distractor", prior.Prior(src, permute=s)) for s in PERMUTATION_SEEDS},
        **{f"R_{s}": ("schema", prior.Prior(src, permute=s)) for s in PERMUTATION_SEEDS},
        "A_noobs": ("schema", prior.Prior(src, prior.NO_OBSERVATION)),
    }


def restriction_key(R):
    parts = [R.m[s].tobytes() for s in sorted(R.m)]
    parts += [R.occ[s].tobytes() + R.abs[s].tobytes() for s in sorted(R.occ)]
    return b"|".join(parts)


def frac(x):
    x = Fraction(x)
    return {"num": str(x.numerator), "den": str(x.denominator), "float": float(x),
            "log10": (math.log10(x.numerator) - math.log10(x.denominator)) if x > 0 else None}


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def score_samples(T48, samples):
    if not samples:
        return None
    rows = np.array([[s[k] for k in SLOTS] for s in samples], dtype=np.int64)
    conform, hits = engine.evaluate_batch(T48, rows)
    train_ok = hits[:, :N_TRAIN].all(axis=1)
    held = hits[:, N_TRAIN:]
    gen = held.all(axis=1)
    return {"n": len(samples), "train_conform_all": bool(train_ok.all()),
            "held_out_accuracy_mean": float(held.mean()),
            "held_out_accuracy_min": float(held.mean(axis=1).min()),
            "generalize_fraction": float(gen.mean()), "generalize_count": int(gen.sum()),
            "generalize_wilson95": wilson(int(gen.sum()), len(samples)),
            "held_out_accuracy_hist": {str(round(a, 4)): int(c) for a, c in
                                       zip(*np.unique(np.round(held.mean(axis=1), 4),
                                                      return_counts=True))}}


def slot_summary(samples, pool, scores):
    out = {}
    for s in SLOTS:
        vals = [x[s] for x in samples]
        top = max(set(vals), key=vals.count)
        entry = {"most_common": str(pool[s][top]), "share": vals.count(top) / len(vals),
                 "distinct": len(set(vals))}
        if scores is not None and KIND[s] != "LIT":
            sc = np.asarray(scores[s])
            entry["prior_rank_of_most_common"] = int((sc > sc[top]).sum()) + 1
            entry["prior_score_of_most_common"] = float(sc[top])
            entry["slot_best_score"] = float(sc.max())
        out[s] = entry
    return out


def run_arm(name, pool_name, pr, gap, n_samples, seed):
    d = cache.load(gap)
    train = engine.Episodes(d["episodes"][:N_TRAIN])
    allep = engine.Episodes(d["episodes"])
    pool = family.pools(pool_name, WIDTH)
    T = engine.Tables(train, pool)
    T48 = engine.Tables(allep, pool)
    scores = prior.slot_scores(pr, pool, train, WIDTH) if pr is not None else None
    tiers = prior.tiers(scores, pool) if pr is not None else [engine.Restriction(pool)]
    rng = random.Random(seed)
    rows, seen = [], {}
    prev_S = prev_K = 0
    before = 0
    first = None
    cost = None
    samples_stats = None
    summary = None
    started = time.perf_counter()
    for t, R in enumerate(tiers):
        key = restriction_key(R)
        t0 = time.perf_counter()
        S = engine.space_size(T, R)
        need_samples = first is None
        if key in seen and not need_samples:
            K, detail = seen[key]
            counter = None
        else:
            counter = engine.Counter(T, R, collect=need_samples)
            K = counter.total
            detail = {"routes": len(counter.route_list), "classes": len(counter.class_list),
                      "matcher_tables": len(counter.tables), "distinct_L": len(counter.rcache)}
            seen[key] = (K, detail)
        shell_S, shell_K = S - prev_S, K - prev_K
        row = {"tier": t, "S": str(S), "S_log10": math.log10(S) if S else None,
               "K": str(K), "shell_S": str(shell_S), "shell_K": str(shell_K),
               "sizes": {s: int(R.m[s].sum()) for s in R.m},
               "lit_occurs_allowed": {s: bool(R.occ[s].any()) for s in R.occ},
               "lit_absent_allowed": {s: bool(R.abs[s].any()) for s in R.abs},
               "certificate": "unique" if K == 1 else "complete", "counter": detail,
               "seconds": time.perf_counter() - t0}
        rows.append(row)
        print(name, row["tier"], "S=%.3e" % S if S else 0, "K=", K, "%.1fs" % row["seconds"],
              flush=True)
        if first is None:
            if shell_K > 0:
                first = t
                cost = Fraction(before) + Fraction(shell_S + 1, shell_K + 1)
                samples = counter.sample(n_samples, rng)
                samples_stats = score_samples(T48, samples)
                summary = slot_summary(samples, pool, scores)
                example = samples[0]
            else:
                before += shell_S
        prev_S, prev_K = S, K
    full_S, full_K = prev_S, prev_K
    out = {"arm": name, "gap": gap, "pool": pool_name, "prior": pr is not None,
           "tiers": rows, "first_solvable_tier": first,
           "full_space": {"S": str(full_S), "S_log10": math.log10(full_S) if full_S else None,
                          "K": str(full_K), "certificate": "unique" if full_K == 1 else "complete",
                          "exhausted": True},
           "expected_programs": frac(cost) if cost is not None else None,
           "expected_episodes": frac(cost * N_TRAIN) if cost is not None else None,
           "solution_exists": full_K > 0,
           "first_solution_samples": samples_stats, "first_solution_slots": summary,
           "example_first_solution": ({s: int(example[s]) for s in SLOTS}
                                      if first is not None else None),
           "example_first_solution_specs": ({s: str(pool[s][example[s]]) for s in SLOTS}
                                            if first is not None else None),
           "seconds": time.perf_counter() - started}
    if pr is not None:
        out["prior_estimators"] = pr.describe()
    (OUT / f"arm_gap{gap}_{name.replace(chr(39), 'p')}.json").write_text(
        json.dumps(out, indent=1, default=str))
    return out


# ------------------------------------------------------------- extras
def hard_vector(pool, gap):
    """PREREGISTRATION §4.4 H, by value where the target pool holds it."""
    def idx(slot, spec):
        try:
            return pool[slot].index(spec)
        except ValueError:
            return 0
    sel = {"cpos": idx("cpos", ("sub", "length", "k1")), "ra": idx("ra", ("sub", "length", "k1")),
           "lc1": idx("lc1", 40), "lc2": idx("lc2", 40), "lc3": idx("lc3", 40),
           "lr1": idx("lr1", 40), "lr2": idx("lr2", 40),
           "K1": 0, "K2": 0, "K3": 0, "K4": 0, "M": idx("M", (1, 2, 7, 2)),
           "X1": idx("X1", ("add", "lo", 3)), "X2": idx("X2", ("add", "lo", 3 * WIDTH)),
           "X3": idx("X3", ("add", "lo", 3 * WIDTH))}
    return sel


def baselines(records):
    train, held = records[:N_TRAIN], records[N_TRAIN:]

    def hits_at(q, r):
        return int(r["owner"][q] == r["target"])
    train_hits = [sum(hits_at(q, r) for r in train) for q in range(engine.N_PIX)]
    best_q = int(np.argmax(train_hits))
    held_hits = [sum(hits_at(q, r) for r in held) for q in range(engine.N_PIX)]
    return {"best_constant_click_train_pixel": best_q,
            "best_constant_click_train_accuracy": train_hits[best_q] / len(train),
            "best_constant_click_held_accuracy": held_hits[best_q] / len(held),
            "oracle_constant_click_held_accuracy": max(held_hits) / len(held),
            "random_pixel_held_accuracy": float(np.mean([sum(o == r["target"] for o in r["owner"])
                                                         / engine.N_PIX for r in held])),
            "random_child_held_accuracy": float(np.mean([1 / sum(w["parent"] == 0 for w in r["widgets"])
                                                         for r in held]))}


def extras(gap):
    d = cache.load(gap)
    allep = engine.Episodes(d["episodes"])
    pool = family.pools("schema", WIDTH)
    T48 = engine.Tables(allep, pool)
    sel = hard_vector(pool, gap)
    conform, hits = engine.evaluate_batch(T48, np.array([[sel[s] for s in SLOTS]]))
    records = d["episodes"]
    achieved = {
        "train_combos": sorted({(r["colour"], r["relation"]) for r in records[:N_TRAIN]}),
        "held_combo_counts": {f"{c}/{r}": sum(1 for x in records[N_TRAIN:]
                                              if x["colour"] == c and x["relation"] == r)
                              for c, r in sorted({(x["colour"], x["relation"]) for x in records})},
        "widgets_per_screen": {str(k): sum(1 for r in records if len(r["widgets"]) == k)
                               for k in sorted({len(r["widgets"]) for r in records})},
        "draws": [r["draws"] for r in records],
        "text_lengths": sorted({r["text_length"] for r in records}),
        "anchor_widths": [r["widgets"][r["anchor"]]["rect"][2] for r in records],
        "anchor_heights": [r["widgets"][r["anchor"]]["rect"][3] for r in records],
        "target_areas": [sum(o == r["target"] for o in r["owner"]) for r in records],
    }
    out = {"gap": gap, "hard_vector": {"selection": sel,
                                       "specs": {s: str(pool[s][sel[s]]) for s in SLOTS},
                                       "train_accuracy": float(hits[0, :N_TRAIN].mean()),
                                       "held_accuracy": float(hits[0, N_TRAIN:].mean()),
                                       "conforms_on_train": bool(hits[0, :N_TRAIN].all())},
           "baselines": baselines(records), "achieved": achieved}
    (OUT / f"extras_gap{gap}.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=int, default=0)
    ap.add_argument("--arm", default="N''")
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.arm == "extras":
        print(json.dumps(extras(a.gap), indent=1, default=str)[:3000])
    else:
        src = prior.load_sources()
        table = arm_table(src)
        pool_name, pr = table[a.arm]
        r = run_arm(a.arm, pool_name, pr, a.gap, a.samples, a.seed)
        print(json.dumps({k: v for k, v in r.items() if k not in ("tiers", "prior_estimators")},
                         indent=1, default=str)[:4000])
