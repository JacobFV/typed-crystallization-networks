"""Sensitivity of the rule's rank-1 proposal to its three declared parameters."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mine
from corpus import EARLIER_TASKS, examples_for
from run_mine import load_corpus

OUT = Path(__file__).parent / "out"


def main():
    depth = sys.argv[1] if len(sys.argv) > 1 else "exact"
    corpus, _ = load_corpus(depth)
    fns = dict(EARLIER_TASKS)
    ex = {k: examples_for(fns[k]) for k in corpus}
    rows = []
    for max_nodes in (2, 3, 4, 5, 6):
        for max_holes in (2, 3, 4, 5):
            for min_tasks in (2, 3):
                props = mine.propose(corpus, ex, max_nodes=max_nodes,
                                     max_holes=max_holes, min_tasks=min_tasks)
                top = props[0] if props else None
                rows.append({
                    "max_nodes": max_nodes, "max_holes": max_holes,
                    "min_tasks": min_tasks, "eligible": len(props),
                    "top_digest": top.digest if top else None,
                    "top_nodes": top.nodes if top else None,
                    "top_holes": top.holes if top else None,
                    "top_saving_bits": top.saving_bits if top else None,
                    "top_tasks": list(top.tasks) if top else None,
                    "top_truth_table": ([[list(b), v] for b, v in
                                         mine.truth_table(top.canonical)]
                                        if top and top.holes <= 4 else None),
                })
                print(json.dumps(rows[-1], sort_keys=True), flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / f"sensitivity_d{depth}.json").write_text(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
