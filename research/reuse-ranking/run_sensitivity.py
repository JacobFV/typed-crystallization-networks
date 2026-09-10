"""Two sensitivity reports, run after the declared arms and tuning nothing.

**A. The breadth exponent.**  O2 is `|T|^1 · Σ s_e − D`; the incumbent O1 is the
same expression at `|T|^0`.  Sweeping the exponent `α` says how knife-edge the
win is.  §52 measured the incumbent's rank-1 decision turning on **0.85 %**;
this is the analogous number for the replacement.

**B. The rule's own parameters.**  `MAX_NODES × MAX_HOLES × MIN_TASKS` over
`{3,4,5} × {2,3,4} × {2,3,4}`, the identical grid §52 §8 swept.  Every declared
arm of this track uses the defaults `5 × 4 × 2`; this grid is reported, not
selected from.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import _paths  # noqa: F401

import context
import objectives
import pool

OUT = Path(__file__).resolve().parent / "out"
LOO = ("t1_maj_abc_xor_d", "t2_maj_abc_and_d", "t3_maj_bcd_or_a",
       "t4_maj_acd_xor_b", "t5_maj_abd_or_c")
CORPORA = [("full", ())] + [(f"wo_{t.split('_')[0]}", (t,)) for t in LOO]


def exponent_sweep(alphas):
    """Is the window class rank 1 under `|T|^α · Σ s_e − D`?"""
    ranked = json.loads((OUT / "ranked.json").read_text())
    res = {}
    for name, _ in CORPORA:
        rows = ranked["corpora"][name]["tables"]["O1"]["ranked"]
        cand = [(r["digest"], len(r["tasks"]), r["saving_bits"] + r["definition_bits"],
                 r["definition_bits"], r["occurrences"], r["nodes"], r["is_window"])
                for r in rows]
        per = {}
        for a in alphas:
            top = min(cand, key=lambda x: (-((x[1] ** a) * x[2] - x[3]), -x[1],
                                           -x[4], -x[5], x[0]))
            per[f"{a:g}"] = {"digest": top[0], "is_window": bool(top[6])}
        win = sorted(float(a) for a, v in per.items() if v["is_window"])
        res[name] = {"per_alpha": per,
                     "window_rank1_alpha_min": win[0] if win else None,
                     "window_rank1_alpha_max": win[-1] if win else None,
                     "window_rank1_in_all_swept": len(win) == len(alphas)}
    return res


def parameter_grid():
    """§52 §8's grid, applied to every objective instead of only to the rule."""
    rows = []
    for mn, mh, mt in itertools.product((3, 4, 5), (2, 3, 4), (2, 3, 4)):
        params = {"max_nodes": mn, "max_holes": mh, "min_tasks": mt}
        for name, exclude in CORPORA:
            t0 = time.perf_counter()
            classes, corpus, task_of, _ = pool.build("maj", "C-minall", exclude,
                                                     keep_rewritten=False, **params)
            constructible = any(pool.is_window(k) for k in classes)
            ctx = context.build_context("maj", "C-minall", exclude, classes, corpus,
                                        task_of, need=("O3", "O4"), params=params)
            rec = {"max_nodes": mn, "max_holes": mh, "min_tasks": mt, "corpus": name,
                   "eligible": len(classes), "window_constructible": constructible,
                   "wall_seconds": time.perf_counter() - t0, "rank1": {}}
            for oid in ("O1", "O2", "O3", "O4", "B1", "B2"):
                tbl, _o = objectives.rank(classes, oid, ctx)
                rec["rank1"][oid] = {"digest": tbl[0]["digest"],
                                     "is_window": bool(tbl[0]["is_window"]),
                                     "window_rank": min([r["rank"] for r in tbl
                                                         if r["is_window"]], default=None)}
            rows.append(rec)
            print(f"{mn}/{mh}/{mt} {name:8s} eligible={len(classes):3d} "
                  f"constructible={constructible} " +
                  " ".join(f"{o}={rec['rank1'][o]['is_window']:d}"
                           for o in ("O1", "O2", "O3", "O4", "B1", "B2")), flush=True)
    return rows


def main():
    alphas = [0, .02, .05, .08, .1, .25, .5, .75, 1, 1.5, 2, 3, 5, 10]
    payload = {"exponent": {"alphas": alphas, "result": exponent_sweep(alphas)}}
    payload["parameter_grid"] = parameter_grid()
    # summarise: of the settings where the window class is constructible at all,
    # in how many is each objective's rank 1 the window class?
    summ = {}
    for oid in ("O1", "O2", "O3", "O4", "B1", "B2"):
        ok = [r for r in payload["parameter_grid"] if r["window_constructible"]]
        summ[oid] = {"of_constructible": len(ok),
                     "rank1_is_window": sum(1 for r in ok if r["rank1"][oid]["is_window"]),
                     "of_all": len(payload["parameter_grid"]),
                     "rank1_is_window_all": sum(1 for r in payload["parameter_grid"]
                                                if r["rank1"][oid]["is_window"])}
    payload["parameter_grid_summary"] = summ
    (OUT / "sensitivity.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps(summ, indent=1, sort_keys=True))
    print("->", OUT / "sensitivity.json")


if __name__ == "__main__":
    main()
