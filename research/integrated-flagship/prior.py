"""The learned specialization prior (PREREGISTRATION §5) and its tiers.

Fitted **only** on `out/sources.json` -- the earlier domains' candidates, each
labelled by whether it survives in some conforming program of its own exhausted
search. Nothing here reads the integrated task's labels. The target task
contributes only *unlabelled* observation statistics (V, occurs), computed the
same way they were computed on the sources.

Estimator: per slot kind, P(survive | feature tuple) with Laplace smoothing
(alpha = 1, prior mean = the kind's overall survival rate), backing off to a
coarser tuple (drop the last feature) when a tuple is unseen, and to the kind
rate when even the coarsest is unseen. Hand-initialised, and ablated in the
arms: the feature set, alpha, and the back-off order.

Tiers: tier t keeps each slot's candidates whose score is >= (the slot's best
score) / 2**t. A LIT candidate's score depends on whether it occurs at the
slot's address, so a LIT tier is a pair of letter masks (occurs / absent).
"""
from __future__ import annotations

import json
import math
import pathlib
import random
from collections import defaultdict

import numpy as np

import engine
from family import KIND, SLOTS

HERE = pathlib.Path(__file__).resolve().parent
ALPHA = 1.0
FEATURES = {"ADDR": ("op", "kinds", "V"), "LIT": ("occurs",), "TRUTH": ("table",),
            "STEP": ("op", "offset_class")}
NO_OBSERVATION = {"ADDR": ("op", "kinds"), "LIT": (), "TRUTH": ("table",),
                  "STEP": ("op", "offset_class")}


def load_sources():
    return json.loads((HERE / "out" / "sources.json").read_text())


def _key(row, feats):
    return tuple(tuple(row[f]) if isinstance(row[f], list) else row[f] for f in feats)


class Estimator:
    def __init__(self, rows, feats, alpha=ALPHA):
        self.feats = feats
        self.alpha = alpha
        self.p0 = (sum(r["survive"] for r in rows) / len(rows)) if rows else None
        self.table = [defaultdict(lambda: [0, 0]) for _ in range(len(feats) + 1)]
        for r in rows:
            for level in range(1, len(feats) + 1):
                cell = self.table[level][_key(r, feats[:level])]
                cell[0] += int(r["survive"])
                cell[1] += 1
        self.n = len(rows)

    def score(self, row):
        if self.p0 is None:
            return 1.0                                   # no data: uniform
        for level in range(len(self.feats), 0, -1):
            cell = self.table[level].get(_key(row, self.feats[:level]))
            if cell is not None and cell[1] > 0:
                return (cell[0] + self.alpha * self.p0) / (cell[1] + self.alpha)
        return self.p0

    def describe(self):
        out = {"p0": self.p0, "rows": self.n, "features": list(self.feats), "cells": {}}
        for level in range(1, len(self.feats) + 1):
            for k, (pos, n) in sorted(self.table[level].items(), key=lambda kv: str(kv[0])):
                out["cells"][str(k)] = {"survive": pos, "n": n,
                                        "score": (pos + self.alpha * self.p0) / (n + self.alpha)}
        return out


class Prior:
    """Four estimators fitted on source rows; `permute` gives the distractor prior."""

    def __init__(self, sources, features=FEATURES, permute=None):
        self.features = features
        self.est = {}
        for kind, feats in features.items():
            rows = [dict(r) for r in sources.get(kind, [])]
            if permute is not None:
                rng = random.Random(permute)
                groups = defaultdict(list)
                for r in rows:
                    groups[(r["source"], r["slot"])].append(r)
                for g in groups.values():
                    labels = [r["survive"] for r in g]
                    rng.shuffle(labels)
                    for r, y in zip(g, labels):
                        r["survive"] = y
            self.est[kind] = Estimator(rows, feats)

    def describe(self):
        return {k: e.describe() for k, e in self.est.items()}


# ------------------------------------------------------- target features
def offset_class(off, width):
    return {3: "pixel", 3 * width: "row", 6: "2pixel", 3 * width + 3: "row+pixel",
            9: "3pixel"}.get(off, "other")


def addr_row(spec, episodes):
    vals = engine.addr_values(spec, episodes.length)
    if (vals == engine.ERR).any():
        v = False
    else:
        v = len({int(episodes.text[e, a]) for e, a in enumerate(vals)}) >= 2
    kinds = sorted("length" if s == "length" else "constant" for s in spec[1:])
    return {"op": spec[0], "kinds": kinds, "V": v}


def slot_scores(prior, pools, episodes, width):
    """Per slot: an array of scores over its pool (LIT: a pair for occurs/absent)."""
    out = {}
    for slot in SLOTS:
        kind = KIND[slot]
        if kind == "ADDR":
            out[slot] = np.array([prior.est["ADDR"].score(addr_row(s, episodes)) for s in pools[slot]])
        elif kind == "LIT":
            out[slot] = (prior.est["LIT"].score({"occurs": True}),
                         prior.est["LIT"].score({"occurs": False}))
        elif kind == "GROUND":
            out[slot] = np.ones(len(pools[slot]))
        elif kind == "MATCH":
            t = prior.est["TRUTH"]
            out[slot] = np.array([t.score({"table": rg}) * t.score({"table": same})
                                  for _, _, rg, same in pools[slot]])
        elif kind == "STEP":
            out[slot] = np.array([prior.est["STEP"].score({"op": op, "offset_class":
                                                           offset_class(off, width)})
                                  for op, _, off in pools[slot]])
    return out


def tiers(scores, pools):
    """Nested restrictions, tier 0 first, ending with the full pool."""
    best = {}
    worst = {}
    for slot, sc in scores.items():
        vals = np.asarray(sc if not isinstance(sc, tuple) else list(sc), dtype=float)
        best[slot] = float(vals.max())
        worst[slot] = float(vals.min())
    t_max = 0
    for slot in scores:
        if worst[slot] <= 0:
            raise ValueError("a zero score would never enter a tier")
        t_max = max(t_max, math.ceil(math.log2(best[slot] / worst[slot]) - 1e-12))
    out = []
    for t in range(t_max + 1):
        masks, occ, ab = {}, {}, {}
        for slot, sc in scores.items():
            thr = best[slot] / (2 ** t) * (1 - 1e-12)
            if KIND[slot] == "LIT":
                n = len(pools[slot])
                occ[slot] = np.full(n, sc[0] >= thr)
                ab[slot] = np.full(n, sc[1] >= thr)
            else:
                masks[slot] = np.asarray(sc) >= thr
        out.append(engine.Restriction(pools, masks, occ, ab))
    return out


def rank_in_slot(scores, slot, index, occurs=None):
    """1-based rank of a candidate in its slot's prior order (ties share the best rank)."""
    sc = scores[slot]
    if KIND[slot] == "LIT":
        mine = sc[0] if occurs else sc[1]
        return 1 if mine >= max(sc) else 2
    return int((np.asarray(sc) > sc[index]).sum()) + 1
