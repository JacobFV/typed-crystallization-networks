"""The rule's three parameters swept over the primary corpora.

Declared in the pre-registration as a reported-regardless robustness check, run
once and reported whatever the headline says -- exactly as §44 did.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mine_multi
import run_mine_premin as R
from corpus import EARLIER_TASKS, examples_for

OUT = Path(__file__).parent / "out"
GRID = [(n, h, t) for n, h, t in itertools.product((2, 3, 4, 5, 6), (2, 3, 4, 5), (2, 3))]


def one(job):
    variant, (max_nodes, max_holes, min_tasks) = job
    corpus, task_of = R.build_variant(variant, R.load_bands())
    fns = dict(EARLIER_TASKS)
    ex = {e: examples_for(fns[task_of[e]]) for e in corpus}
    t0 = time.perf_counter()
    props = mine_multi.propose(corpus, ex, task_of, max_nodes=max_nodes,
                               max_holes=max_holes, min_tasks=min_tasks)
    maj = [p for p in props if R.is_maj(p)]
    top = props[0] if props else None
    return {"variant": variant, "max_nodes": max_nodes, "max_holes": max_holes,
            "min_tasks": min_tasks, "eligible": len(props),
            "rank1_digest": (top.digest if top else None),
            "rank1_nodes": (top.nodes if top else None),
            "rank1_holes": (top.holes if top else None),
            "rank1_saving_bits": (top.saving_bits if top else None),
            "rank1_body": (R.body_of(top) if top else None),
            "rank1_computes_maj3": bool(top and R.is_maj(top)),
            "best_maj3_rank": (min(p.rank for p in maj) if maj else None),
            "wall_seconds": time.perf_counter() - t0}


def main():
    variants = sys.argv[1:] or ["C-trace", "C-minall"]
    jobs = [(v, g) for v in variants for g in GRID]
    with ProcessPoolExecutor(max_workers=8) as pool:
        recs = list(pool.map(one, jobs))
    OUT.mkdir(exist_ok=True)
    (OUT / "sensitivity.json").write_text(json.dumps({"grid": GRID, "records": recs},
                                                     indent=2, sort_keys=True))
    for v in variants:
        rows = [r for r in recs if r["variant"] == v]
        digests = {}
        for r in rows:
            digests[r["rank1_digest"]] = digests.get(r["rank1_digest"], 0) + 1
        print(v, len(rows), "settings;", "rank1 digests:",
              {(k[:12] if k else None): n for k, n in sorted(digests.items(),
                                                             key=lambda kv: -kv[1])})
        print("   settings where rank1 computes maj3:",
              sum(1 for r in rows if r["rank1_computes_maj3"]),
              "; best maj3 ranks:",
              sorted({r["best_maj3_rank"] for r in rows}, key=lambda x: (x is None, x)))


if __name__ == "__main__":
    main()
