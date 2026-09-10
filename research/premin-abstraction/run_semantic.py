"""Run the semantic-identity variant of the rule, per corpus. Exploratory.

Not pre-registered. Measures how much of this track's negative is caused by
R2's structural identity relation rather than by the corpus, by re-running the
identical rule with occurrences pooled by (arity, truth table) instead of by
`Program.digest`.  Publishes the rank-1 proposal for `C-trace` so it can be
inherited through the same library path.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

from tcn.library import Library
from tcn.operators import Registry
from tcn.types import BOOL, Value

import corpora
import mine_semantic
import retention
import run_mine_premin as R
from corpus import EARLIER_TASKS, examples_for

HERE = Path(__file__).parent
OUT = HERE / "out"
LIB = HERE / "library"


def body_of(canon):
    return [[n.candidates[0].operator.name, n.name, list(n.candidates[0].sources)]
            for n in canon.nodes]


def main():
    variants = sys.argv[1:] or ["C-min", "C-minall", "C-plus1", "C-trace", "C-plus1-one"]
    bands = R.load_bands()
    fns = dict(EARLIER_TASKS)
    summary = {}
    for variant in variants:
        corpus, task_of = R.build_variant(variant, bands)
        ex = {e: examples_for(fns[task_of[e]]) for e in corpus}
        props = mine_semantic.propose(corpus, ex, task_of)
        table = []
        for p in props:
            row = p.row()
            row["body"] = body_of(p.canonical)
            row["computes_maj3"] = (p.holes == 3 and p.key[1] == retention.MAJ3_TABLE)
            table.append(row)
        maj = [r for r in table if r["computes_maj3"]]
        payload = {"variant": variant, "identity": "(arity, truth table)",
                   "entries": len(corpus), "eligible": len(props),
                   "rank1_computes_maj3": bool(table and table[0]["computes_maj3"]),
                   "best_maj3_rank": (min(r["rank"] for r in maj) if maj else None),
                   "ranked": table}
        (OUT / f"semantic_{variant}.json").write_text(json.dumps(payload, indent=2,
                                                                 sort_keys=True))
        summary[variant] = {k: payload[k] for k in
                            ("entries", "eligible", "rank1_computes_maj3", "best_maj3_rank")}
        head = table[0] if table else None
        print(variant, "entries", len(corpus), "eligible", len(props),
              "rank1 maj3", payload["rank1_computes_maj3"],
              "rank1", (head["body"] if head else None),
              "saving", (head["saving_bits"] if head else None),
              "pooled circuits", (head["circuits_pooled"] if head else None),
              "best maj3 rank", payload["best_maj3_rank"], flush=True)

    if "C-trace" in variants:
        d = json.loads((OUT / "semantic_C-trace.json").read_text())
        top = d["ranked"][0]
        corpus, _ = R.build_variant("C-trace", bands)
        canon = None
        for _e, program in corpus.items():
            for _r, _S, c, _h in R.mine.fragments(program, mine_semantic.MAX_NODES,
                                                  mine_semantic.MAX_HOLES):
                if c.digest == top["digest"]:
                    canon = c
                    break
            if canon is not None:
                break
        assert canon is not None
        fixture = [{k: Value.of(BOOL, v) for (k, _), v in zip(canon.inputs, bits)}
                   for bits in itertools.product((False, True), repeat=len(canon.inputs))]
        entry = Library(LIB).publish("semantic_trace", canon, Registry(), fixture=fixture,
                                     provenance={"rule": "mine_semantic.propose",
                                                 "corpus": "C-trace", "rank": top["rank"],
                                                 "tasks": top["tasks"],
                                                 "entries": top["entries"],
                                                 "circuits_pooled": top["circuits_pooled"],
                                                 "saving_bits": top["saving_bits"],
                                                 "exploratory": True})
        summary["published"] = {"reference": entry.reference, "operator": entry.operator,
                                "nodes": entry.nodes, "digest": entry.digest,
                                "computes_maj3": top["computes_maj3"]}
        print("published", entry.reference, entry.operator, entry.nodes, "nodes",
              "maj3", top["computes_maj3"])
    (OUT / "semantic_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
