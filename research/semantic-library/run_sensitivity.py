"""The rule's three parameters swept once, under semantic identity, and reported
whatever the headline says.  The rule is not re-tuned until a number improves.
"""
from __future__ import annotations

import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _paths  # noqa: F401

import mine_semantic
import retention

import evaltasks
import family

OUT = Path(__file__).resolve().parent / "out"
GRID = list(itertools.product((3, 4, 5), (2, 3, 4), (2, 3, 4)))


def one(cfg):
    max_nodes, max_holes, min_tasks = cfg
    corpus, task_of = family.corpus_for("maj", "C-minall")
    fns = dict(family.MAJ_TASKS)
    ex = {e: evaltasks.examples_for(fns[task_of[e]]) for e in corpus}
    t0 = time.perf_counter()
    props = mine_semantic.propose(corpus, ex, task_of, max_nodes=max_nodes,
                                  max_holes=max_holes, min_tasks=min_tasks)
    wall = time.perf_counter() - t0
    win = [p for p in props if p.holes == 3 and p.key[1] == retention.MAJ3_TABLE]
    top = props[0] if props else None
    return {"max_nodes": max_nodes, "max_holes": max_holes, "min_tasks": min_tasks,
            "eligible": len(props), "wall_seconds": wall,
            "rank1_is_maj3": bool(top is not None and top.holes == 3
                                  and top.key[1] == retention.MAJ3_TABLE),
            "best_maj3_rank": (min(p.rank for p in win) if win else None),
            "rank1_arity": (top.holes if top else None),
            "rank1_nodes": (top.nodes if top else None),
            "rank1_digest": (top.digest if top else None)}


def main():
    with ProcessPoolExecutor(max_workers=int(os.environ.get("TCN_WORKERS", "9"))) as pool:
        rows = list(pool.map(one, GRID))
    hit = sum(1 for r in rows if r["rank1_is_maj3"])
    payload = {"grid": len(rows), "rank1_is_maj3": hit, "rows": rows,
               "default": {"max_nodes": mine_semantic.MAX_NODES,
                           "max_holes": mine_semantic.MAX_HOLES,
                           "min_tasks": mine_semantic.MIN_TASKS}}
    OUT.mkdir(exist_ok=True)
    (OUT / "sensitivity.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    for r in rows:
        print(r["max_nodes"], r["max_holes"], r["min_tasks"], "eligible", r["eligible"],
              "rank1_is_maj3", r["rank1_is_maj3"], "best_maj3_rank", r["best_maj3_rank"])
    print("rank1 is MAJ3 in", hit, "of", len(rows), "settings")


if __name__ == "__main__":
    main()
