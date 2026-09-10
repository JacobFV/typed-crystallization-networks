"""Two sensitivity sweeps, both declared in `PREREGISTRATION.md` §6 and §9.

**The breadth exponent.**  §54 swept `|T(c)|^alpha . sum s_e - D` and found the
right class held only for `alpha in [0.08, 3]`, with `alpha = 0` (the
incumbent) and `alpha = 10` (the frequency baseline's class) both failing --
its evidence that *neither term alone suffices*.  The same sweep is run here.
If the window is rank 1 at `alpha = 0` on this family then the breadth term is
doing nothing, which is falsification F3.

**The rule's parameters.**  `MIN_TASKS` and `MAX_NODES` are swept as §52 §8
swept them.  `MAX_HOLES` is not swept below 4: the window has arity 4, so a
lower ceiling removes it from the eligible set by construction, and that is
reported rather than run.  Every declared arm uses the defaults
`MAX_NODES=5, MAX_HOLES=4, MIN_TASKS=2`; nothing is tuned.
"""
from __future__ import annotations

import json
from pathlib import Path

import _paths  # noqa: F401
import patches  # noqa: F401

import context
import objectives
import pool

import evaltasks

OUT = Path(__file__).resolve().parent / "out"

ALPHAS = (0.0, 0.02, 0.05, 0.08, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0)
BANDS = ("C-trace", "C-minall")


def alpha_rank1(classes, alpha, family_name):
    scored = []
    for k in classes:
        s = (len(k.tasks) ** alpha) * sum(k.saving_by_entry.values()) - k.definition_bits
        scored.append((s, k))
    scored.sort(key=lambda sk: (-sk[0], -len(sk[1].tasks), -sk[1].occurrences,
                                -sk[1].nodes, sk[1].digest))
    top = scored[0][1]
    return {"alpha": alpha, "digest": top.digest, "arity": top.holes,
            "nodes": top.nodes, "tasks": len(top.tasks),
            "is_window": pool.is_window(top, family_name),
            "window_rank": next((i + 1 for i, (_s, k) in enumerate(scored)
                                 if pool.is_window(k, family_name)), None)}


def main():
    OUT.mkdir(exist_ok=True)
    corpora = [("full", ())] + [(f"wo_{t.split('_')[0]}", (t,))
                                for t in evaltasks.LOO_TASKS]
    payload = {"alpha_sweep": {}, "params": {}}
    for band in BANDS:
        payload["alpha_sweep"][band] = {}
        for name, exclude in corpora:
            classes, _c, _t, _b = pool.build("w4", band, exclude)
            rows = [alpha_rank1(classes, a, "w4") for a in ALPHAS]
            payload["alpha_sweep"][band][name] = rows
            hold = [r["alpha"] for r in rows if r["is_window"]]
            print(f"{band:9s} {name:8s} window rank1 at alpha in "
                  f"{hold if hold else 'NONE'}", flush=True)

        payload["params"][band] = []
        for min_tasks in (2, 3):
            for max_nodes in (4, 5):
                classes, corpus, task_of, _ = pool.build(
                    "w4", band, (), keep_rewritten=True,
                    max_nodes=max_nodes, min_tasks=min_tasks)
                ctx = context.build_context(
                    "w4", band, (), classes, corpus, task_of, need=(),
                    params={"max_nodes": max_nodes, "min_tasks": min_tasks})
                cell = {"min_tasks": min_tasks, "max_nodes": max_nodes,
                        "max_holes": 4, "eligible": len(classes)}
                for oid in ("O1", "O2", "B1"):
                    rows, _ = objectives.rank(classes, oid, ctx)
                    cell[oid] = {"digest": rows[0]["digest"],
                                 "is_window": rows[0]["is_window"],
                                 "window_rank": min([r["rank"] for r in rows
                                                     if r["is_window"]],
                                                    default=None),
                                 "decided_by": objectives.rank1_decided_by(rows)}
                payload["params"][band].append(cell)
                print(f"{band:9s} min_tasks={min_tasks} max_nodes={max_nodes} "
                      f"eligible={len(classes):3d} "
                      + " ".join(f"{o}={'WIN' if cell[o]['is_window'] else 'no'}"
                                 f"(r{cell[o]['window_rank']})"
                                 for o in ("O1", "O2", "B1")), flush=True)
    (OUT / "sensitivity.json").write_text(json.dumps(payload, indent=1,
                                                     sort_keys=True))
    print("->", OUT / "sensitivity.json")


if __name__ == "__main__":
    main()
