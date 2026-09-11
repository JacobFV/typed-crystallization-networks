"""The arms: order the same edit space seven ways and cost each exactly.

Reads `out/cases_<domain>.json` only.  Every arm is a **re-ordering** of one
case's edit list, so no arm can make a repair unreachable and the distractor of
C4 can succeed by construction (checked as V3).  Costs are the exact tiered
expectation of PREREGISTRATION section 6.1; nothing is sampled.

    python analyse.py            ->  out/analysis.json
"""
from __future__ import annotations

import itertools
import json
import math
import random
from fractions import Fraction

import kit

ALPHA = 1.0
DOMAINS = ["bool", "arith", "rel"]
ARMS = ["N", "N'", "N''", "H1", "H2", "D1", "D2", "ORACLE"]


# ------------------------------------------------------------------ features

def bucket_added(n):
    for hi, name in ((0, "0"), (4, "1-4"), (16, "5-16"), (48, "17-48")):
        if n <= hi:
            return name
    return ">48"


def bucket_space(x):
    for hi in (3, 4, 5, 6):
        if x < hi:
            return f"<{hi}"
    return ">=6"


def discretize(f):
    return {
        "family": f["family"],
        "op_class": f["op_class"],
        "novel": str(f["novel_operator"]),
        "is_output": str(f["site_is_output"]),
        "const_site": str(f["site_constant_on_train"]),
        "added": bucket_added(f["added_candidates"]),
        "space": bucket_space(f["log_space"]),
        "arity": str(f["max_arity"]),
    }


FEATURE_KEYS = ("family", "op_class", "novel", "is_output", "const_site",
                "added", "space", "arity")


# ------------------------------------------------------------------ estimator

def fit(rows, labels):
    """Additive per-feature empirical log-odds of repair, Laplace-smoothed."""
    values = {k: sorted({r[k] for r in rows}) for k in FEATURE_KEYS}
    pos = [r for r, y in zip(rows, labels) if y]
    neg = [r for r, y in zip(rows, labels) if not y]
    table = {}
    for k in FEATURE_KEYS:
        m = len(values[k])
        for v in values[k]:
            p = (sum(1 for r in pos if r[k] == v) + ALPHA) / (len(pos) + ALPHA * m)
            q = (sum(1 for r in neg if r[k] == v) + ALPHA) / (len(neg) + ALPHA * m)
            table[f"{k}={v}"] = round(math.log(p) - math.log(q), 6)
    return {"table": table, "n_pos": len(pos), "n_neg": len(neg),
            "values": values,
            "flat": all(abs(x) < 1e-9 for x in table.values())}


def score(est, row):
    return round(sum(est["table"].get(f"{k}={row[k]}", 0.0) for k in FEATURE_KEYS), 6)


# ---------------------------------------------------------------------- cost

def expected_cost(tiers):
    """Exact expected edits enumerated before the first repair (section 6.1)."""
    before = 0
    for s, k in tiers:
        if k > 0:
            return Fraction(before) + Fraction(s + 1, k + 1)
        before += s
    return None                      # no repair anywhere in the ordering


def tiers_from_keys(keys, repairs):
    """Group by key in the given (already sorted) order -> [(size, repairs)]."""
    out = []
    for _, grp in itertools.groupby(range(len(keys)), key=lambda i: keys[i]):
        idx = list(grp)
        out.append((len(idx), sum(1 for i in idx if repairs[i])))
    return out


# ---------------------------------------------------------------------- arms

def arm_orders(edits, est, dist_est, family_class):
    """A sort key per edit, per arm.  A constant key means one uniform tier."""
    rows = [discretize(e["features"]) for e in edits]
    keys = {}
    keys["N"] = [0] * len(edits)
    keys["N'"] = [0 if e["family"] in family_class else 1 for e in edits]
    keys["N''"] = [-score(est, r) for r in rows]
    keys["H1"] = [(e["features"]["added_candidates"],
                   0 if e["features"]["novel_operator"] else 1) for e in edits]
    keys["H2"] = [-e["features"]["site_depth_frac"] for e in edits]
    keys["D1"] = [-score(dist_est, r) for r in rows]
    keys["D2"] = [score(est, r) for r in rows]
    keys["ORACLE"] = [0 if e["repair"] else 1 for e in edits]
    return keys


def cost_for(edits, keys, optimistic):
    reps = [(True if e["repair"] is None and optimistic else bool(e["repair"]))
            for e in edits]
    order = sorted(range(len(edits)), key=lambda i: (keys[i], i))
    return expected_cost(tiers_from_keys([keys[i] for i in order],
                                         [reps[i] for i in order]))


def deployed(edits, keys):
    """M2: stop at the first edit with a *training* conformer; is it a repair?"""
    order = sorted(range(len(edits)), key=lambda i: (keys[i], i))
    for rank, i in enumerate(order, start=1):
        if edits[i]["train_conforming"]:
            return {"stopped_at": rank, "repair": bool(edits[i]["repair"]),
                    "key": edits[i]["key"]}
    return {"stopped_at": None, "repair": False, "key": None}


# ---------------------------------------------------------------------- main

def load_domain(name):
    d = kit.load(f"cases_{name}")
    d["admitted_cases"] = [c for c in d["cases"] if c.get("admitted")]
    return d


def main():
    kit.check_workers(1)
    data = {n: load_domain(n) for n in DOMAINS if (kit.OUT / f"cases_{n}.json").exists()}
    present = sorted(data)
    out = {"domains": present, "arms": ARMS, "alpha": ALPHA, "per_domain": {},
           "cases": [], "estimators": {}, "validity": {}}

    # rows per domain, decided edits only (labels must exist to fit on them)
    rows_by_domain, labels_by_domain, fams_by_domain = {}, {}, {}
    for n in present:
        rows, labels, fams = [], [], set()
        for c in data[n]["admitted_cases"]:
            for e in c["edits"]:
                if e["repair"] is None:
                    continue
                rows.append(discretize(e["features"]))
                labels.append(bool(e["repair"]))
                if e["repair"]:
                    fams.add(e["family"])
        rows_by_domain[n], labels_by_domain[n], fams_by_domain[n] = rows, labels, fams

    rng = random.Random(0)
    for held in present:
        src = [n for n in present if n != held]
        if not src:
            continue
        rows = [r for n in src for r in rows_by_domain[n]]
        labels = [y for n in src for y in labels_by_domain[n]]
        if not rows or not any(labels) or all(labels):
            out["estimators"][held] = {"error": "source rows carry no both-class signal"}
            continue
        est = fit(rows, labels)
        family_class = sorted(set().union(*(fams_by_domain[n] for n in src)))
        # D1: the same estimator on permuted labels, 20 permutations, seed 0
        perms = []
        for _ in range(20):
            p = labels[:]
            rng.shuffle(p)
            perms.append(fit(rows, p))
        out["estimators"][held] = {
            "sources": src, "n_pos": est["n_pos"], "n_neg": est["n_neg"],
            "flat": est["flat"], "family_class": family_class,
            "table": est["table"],
            "top_features": sorted(est["table"].items(), key=lambda kv: -abs(kv[1]))[:12],
        }

        for c in data[held]["admitted_cases"]:
            edits = c["edits"]
            keys = arm_orders(edits, est, perms[0], family_class)
            row = {"domain": held, "case_id": c["case_id"], "defect": c["defect"],
                   "n_edits": c["n_edits"], "n_repairs": c["n_repairs"],
                   "n_undecided": c["n_undecided"],
                   "n_train_conforming": c["n_train_conforming"],
                   "inverse_repairs": sum(1 for e in edits
                                          if e["repair"] and e["is_inverse"]),
                   "repair_families": sorted({e["family"] for e in edits if e["repair"]}),
                   "cost": {}, "cost_optimistic": {}, "deployed": {}}
            for a in ARMS:
                row["cost"][a] = cost_for(edits, keys[a], False)
                row["cost_optimistic"][a] = cost_for(edits, keys[a], True)
                row["deployed"][a] = deployed(edits, keys[a])
            # D1 spread over the 20 permutations
            rows_d = [discretize(e["features"]) for e in edits]
            d1 = [cost_for(edits, [-score(p, rr) for rr in rows_d], False)
                  for p in perms]
            row["D1_permutations"] = [str(x) for x in d1]
            row["D1_mean"] = str(sum(d1) / len(d1))
            for k in ("cost", "cost_optimistic"):
                row[k] = {a: (str(v) if v is not None else None)
                          for a, v in row[k].items()}
            out["cases"].append(row)

    # per-domain and overall means
    def mean_cost(rows, arm, key="cost"):
        vals = [Fraction(r[key][arm]) for r in rows if r[key][arm] is not None]
        return (sum(vals) / len(vals)) if vals else None

    for n in present:
        rows = [r for r in out["cases"] if r["domain"] == n]
        if not rows:
            continue
        out["per_domain"][n] = {
            "n_cases": len(rows),
            "n_edits_mean": round(sum(r["n_edits"] for r in rows) / len(rows), 2),
            "n_repairs_mean": round(sum(r["n_repairs"] for r in rows) / len(rows), 2),
            "mean_cost": {a: _f(mean_cost(rows, a)) for a in ARMS},
            "mean_cost_optimistic": {a: _f(mean_cost(rows, a, "cost_optimistic"))
                                     for a in ARMS},
            "deployed_repairs": {a: sum(1 for r in rows if r["deployed"][a]["repair"])
                                 for a in ARMS},
            "inverse_fraction": round(
                sum(r["inverse_repairs"] for r in rows) /
                max(1, sum(r["n_repairs"] for r in rows)), 4),
            "undecided_fraction": round(
                sum(r["n_undecided"] for r in rows) /
                max(1, sum(r["n_edits"] for r in rows)), 4),
        }
    allrows = out["cases"]
    out["overall"] = {
        "n_cases": len(allrows),
        "mean_cost": {a: _f(mean_cost(allrows, a)) for a in ARMS},
        "mean_cost_optimistic": {a: _f(mean_cost(allrows, a, "cost_optimistic"))
                                 for a in ARMS},
        "deployed_repairs": {a: sum(1 for r in allrows if r["deployed"][a]["repair"])
                             for a in ARMS},
        "deployed_of": len(allrows),
    }
    # --- A12: criteria are resolved PER DOMAIN; any pooled figure is the
    # unweighted mean of the per-domain means (macro), never the per-case mean.
    def criteria_from(m):
        if not m or m.get("N''") is None:
            return None
        c = {
            "C1_N''_le_half_N": _crit(m, "N''", "N", 2.0),
            "C2_N''_le_half_Nprime": _crit(m, "N''", "N'", 2.0),
            "C3a_N''_le_half_H1": _crit(m, "N''", "H1", 2.0),
            "C3b_N''_le_half_H2": _crit(m, "N''", "H2", 2.0),
            "C4_D1_fails_C1": (None if m.get("D1") is None or m.get("N") is None
                               else not (m["D1"] <= m["N"] / 2.0)),
        }
        c["met"] = all(c[k] is True for k in
                       ("C1_N''_le_half_N", "C2_N''_le_half_Nprime",
                        "C3a_N''_le_half_H1", "C3b_N''_le_half_H2",
                        "C4_D1_fails_C1"))
        c["ratios"] = {f"{a}/N''": (round(m[a] / m["N''"], 4)
                                    if m.get(a) and m.get("N''") else None)
                       for a in ARMS if a != "N''"}
        return c

    out["criteria_by_domain"] = {}
    for n in present:
        pd = out["per_domain"].get(n)
        if not pd:
            continue
        out["criteria_by_domain"][n] = {
            "n_cases": pd["n_cases"],
            "criteria": criteria_from(pd["mean_cost"]),
            "criteria_optimistic": criteria_from(pd["mean_cost_optimistic"]),
        }

    def macro(key="mean_cost"):
        """Unweighted mean of the per-domain means (A12's pooling rule)."""
        acc = {}
        for a in ARMS:
            vals = [out["per_domain"][n][key][a] for n in present
                    if n in out["per_domain"] and out["per_domain"][n][key][a] is not None]
            acc[a] = (sum(vals) / len(vals)) if vals else None
        return acc

    if allrows:
        out["overall"]["mean_cost_macro"] = macro("mean_cost")
        out["overall"]["mean_cost_macro_optimistic"] = macro("mean_cost_optimistic")
        out["overall"]["pooling_rule"] = (
            "equal weight per domain (macro): the unweighted mean of the "
            "per-domain mean costs, per amendment A12.  `mean_cost` is the "
            "per-case (micro) mean, reported beside it and never used for a "
            "criterion.")
        out["criteria"] = criteria_from(out["overall"]["mean_cost_macro"]) or {}
        mo = out["overall"]["mean_cost_macro_optimistic"]
        co = criteria_from(mo) or {}
        out["criteria"]["met_optimistic"] = co.get("met")
        out["criteria"]["inconclusive"] = (
            out["criteria"].get("met") != co.get("met"))
        # A12(4): a split verdict is NOT MET, and is reported as such.
        per = {n: v["criteria"]["met"] for n, v in out["criteria_by_domain"].items()
               if v["criteria"] is not None}
        out["criteria"]["met_per_domain"] = per
        out["criteria"]["split_verdict"] = (len(set(per.values())) > 1)
        out["criteria"]["met_all_domains"] = bool(per) and all(per.values())

    out["validity"] = {
        "V1_inverse_fraction_overall": round(
            sum(r["inverse_repairs"] for r in allrows) /
            max(1, sum(r["n_repairs"] for r in allrows)), 4) if allrows else None,
        "V3_distractor_space_equals_N''_space": True,
        "V3_note": "every arm re-orders the identical edit list; repairs are "
                   "therefore identical by construction and equal in count",
        "V3_repairs_per_case": {r["case_id"]: r["n_repairs"] for r in allrows},
    }
    print("wrote", kit.dump("analysis", out))
    if allrows:
        print(json.dumps({"overall": out["overall"], "criteria": out["criteria"]},
                         indent=1, default=str))


def _f(x):
    return None if x is None else float(x)


def _crit(m, a, b, factor):
    if m.get(a) is None or m.get(b) is None:
        return None
    return bool(m[a] <= m[b] / factor)


if __name__ == "__main__":
    main()
