"""POST-HOC mechanism table: does generalization track what the prior prefers?

Coordinator-requested, not pre-registered. For every prior-bearing arm at gap 0:

* LIT ratio    score(occurs=True) / score(occurs=False), from the stored estimator;
* ADDR V ratio mean V=True cell score / mean V=False cell score (the crude
  statistic), and -- faithful to how `prior.slot_scores` orders candidates --
* the rank of the correct colour address sub(length, 5) in `cpos`, the best rank
  of any relation address that evaluates to 13 in `ra`, and the tier at which
  absent letters first enter,

beside the arm's sampled first-solution generalize_count and expected cost, with
Spearman correlations over the schema-pool arms.

    python mechanism.py
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np          # noqa: E402

OUT = HERE / "out"
ARMS = ["N''", "P"] + [f"R_{s}" for s in range(5)] + [f"D''_{s}" for s in range(5)] + \
       ["U_feat", "U_steep", "U_Vonly", "KO_ADDR", "KO_TRUTH", "KO_STEP", "STEP_only", "STEP_hand", "A_noobs"] + [f"OCC_{r}" for r in (1, 3, 10, 100, 3500)] + \
       [f"V_{r}" for r in (1, 3, 10, 100)]


def lit_ratio(est):
    c = est["LIT"].get("cells") or {}
    if "(True,)" in c:
        return c["(True,)"]["score"] / c["(False,)"]["score"]
    return 2.0 if "occurs" in est["LIT"].get("rule", "") else 1.0


def v_means(est):
    c = est["ADDR"].get("cells") or {}
    on = [v["score"] for k, v in c.items() if k.endswith(", True)")]
    off = [v["score"] for k, v in c.items() if k.endswith(", False)")]
    if not on or not off:
        return None, None
    return sum(on) / len(on), sum(off) / len(off)


def slot_ranks(arm):
    """Ranks in the arm's own slot order, recomputed through prior.slot_scores."""
    import arms
    import cache
    import engine
    import family
    import prior
    pool_name, pr = arms.arm_table(prior.load_sources())[arm]
    pool = family.pools(pool_name)
    train = engine.Episodes(cache.load(0)["episodes"][:12])
    sc = prior.slot_scores(pr, pool, train, 16)
    cp = np.asarray(sc["cpos"])
    i = pool["cpos"].index(("sub", "length", "k5")) if ("sub", "length", "k5") in pool["cpos"] else None
    ra = np.asarray(sc["ra"])
    thirteen = [j for j, s in enumerate(pool["ra"])
                if (engine.addr_values(s, train.length) == 13).all()]
    lit_on, lit_off = sc["lc1"]
    return {"colour_address_rank": int((cp > cp[i]).sum()) + 1 if i is not None else None,
            "colour_address_score_over_best": float(cp[i] / cp.max()) if i is not None else None,
            "relation_address_best_rank": int(min((ra > ra[j]).sum() + 1 for j in thirteen)) if thirteen else None,
            "absent_letters_enter_tier": math.ceil(math.log2(max(lit_on, lit_off) / lit_off) - 1e-12)}


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    if len(xs) < 3:
        return None
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def main():
    rows = []
    for a in ARMS:
        p = OUT / f"arm_gap0_{a.replace(chr(39), 'p')}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        est = d["prior_estimators"]
        on, off = v_means(est)
        s = d.get("first_solution_samples") or {}
        row = {"arm": a, "pool": d["pool"], "lit_occurs_ratio": lit_ratio(est),
               "addr_V_mean_true": on, "addr_V_mean_false": off,
               "generalize_count": s.get("generalize_count"), "n": s.get("n"),
               "wilson95": s.get("generalize_wilson95"),
               "expected_programs_log10": (d["expected_programs"] or {}).get("log10"),
               "first_solvable_tier": d["first_solvable_tier"]}
        if d["pool"] == "schema":
            row.update(slot_ranks(a))
        rows.append(row)
    schema = [r for r in rows if r["pool"] == "schema" and r["generalize_count"] is not None]
    out = {"rows": rows, "n_schema_arms": len(schema),
           "spearman_log_lit_ratio_vs_generalize": spearman(
               [math.log(r["lit_occurs_ratio"]) for r in schema], [r["generalize_count"] for r in schema]),
           "spearman_colour_rank_vs_generalize": spearman(
               [-r["colour_address_rank"] for r in schema], [r["generalize_count"] for r in schema])}
    (OUT / "mechanism_gap0.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        print(r)
    print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()
