"""Print O4's per-fold decomposition, and write it to `out/o4_folds.json`.

For each leave-one-out corpus and each class of interest: the class's saving on
each fold's *withheld* task -- entries that were removed from the pool that
elected the class and from the score that ranked it.
"""
from __future__ import annotations

import json
from pathlib import Path

import _paths  # noqa: F401

import context
import pool

OUT = Path(__file__).resolve().parent / "out"
LOO = ("t1_maj_abc_xor_d", "t2_maj_abc_and_d", "t3_maj_bcd_or_a",
       "t4_maj_acd_xor_b", "t5_maj_abd_or_c")
LABEL = {"165bc290d9c82b70a8ea3cc2": "window MAJ3 (O2/O4 rank-1)",
         "08735e504400bf4879acf0a7": "O1 rank-1 on wo_t1 / wo_t4 (4-ary, 2 tasks)",
         "0ba4287a0409ec27ba9ef4f2": "O1 rank-1 on wo_t2 / wo_t3 / wo_t5 (4-ary, 2 tasks)",
         "ca785ec948250acbf39b8f66": "pooled runner-up (3-ary, 2 nodes)",
         "82b93e4bc7b294ff04c8f289": "B1 rank-1 (the most task-frequent class)"}


def main():
    out = {}
    for held in LOO:
        name = "wo_" + held.split("_")[0]
        classes, corpus, task_of, _ = pool.build("maj", "C-minall", (held,))
        ctx = context.build_context("maj", "C-minall", (held,), classes, corpus,
                                    task_of, need=("O3", "O4"))
        tasks = ctx["tasks_in_corpus"]
        rows = []
        for k in classes:
            per = []
            for u in tasks:
                hit = ctx["cv"].get(u, {}).get(k.key)
                per.append(0 if hit is None else hit[0] - hit[1])
            rows.append({"digest": k.digest, "label": LABEL.get(k.digest),
                         "tasks": len(k.tasks), "arity": k.holes, "nodes": k.nodes,
                         "saving_bits": k.saving_bits,
                         "folds": dict(zip([t.split("_")[0] for t in tasks], per)),
                         "o4": sum(per) / len(per),
                         "is_window": pool.is_window(k)})
        rows.sort(key=lambda r: -r["o4"])
        out[name] = {"mining_tasks": [t.split("_")[0] for t in tasks], "rows": rows}
        print(f"== {name}  folds over {out[name]['mining_tasks']}")
        for r in rows[:3] + [x for x in rows[3:] if x["label"]]:
            print(f"   {r['digest'][:12]} tasks={r['tasks']} sav={r['saving_bits']:8d} "
                  f"folds={[r['folds'][t] for t in out[name]['mining_tasks']]} "
                  f"O4={r['o4']:.1f}  {r['label'] or ''}")
    (OUT / "o4_folds.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print("->", OUT / "o4_folds.json")


if __name__ == "__main__":
    main()
