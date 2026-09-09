"""Run the selection rule over the solved corpus and publish what it proposes.

Writes the full ranked table (not just the winner), the proposed module, the
runner-up used as the mined wrong-module control, and a real
`tcn.library.Library` publication so the later task inherits through the
shipped publish/inherit path rather than through a Python variable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from tcn.graph import Program
from tcn.operators import Registry
from tcn.library import Library
from tcn.types import BOOL, Value

import mine
from corpus import EARLIER_TASKS, examples_for

HERE = Path(__file__).parent
OUT = HERE / "out"
LIB = HERE / "library"


def load_corpus(depth):
    name = "corpus_exact.json" if depth in ("exact", 0) else f"corpus_d{depth}.json"
    d = json.loads((OUT / name).read_text())
    corpus = {k: Program.from_dict(v, Registry()) for k, v in d["programs"].items()}
    return corpus, d


def main():
    depth = sys.argv[1] if len(sys.argv) > 1 else "exact"
    max_nodes = int(sys.argv[2]) if len(sys.argv) > 2 else mine.MAX_NODES
    max_holes = int(sys.argv[3]) if len(sys.argv) > 3 else mine.MAX_HOLES
    corpus, meta = load_corpus(depth)
    fns = dict(EARLIER_TASKS)
    ex = {k: examples_for(fns[k]) for k in corpus}

    props = mine.propose(corpus, ex, max_nodes=max_nodes, max_holes=max_holes)
    table = []
    for p in props:
        row = p.row()
        row["truth_table"] = [[list(b), v] for b, v in mine.truth_table(p.canonical)] \
            if p.holes <= 4 else None
        row["body"] = [[n.candidates[0].operator.name, n.name,
                        list(n.candidates[0].sources)] for n in p.canonical.nodes]
        table.append(row)

    payload = {"depth": depth, "max_nodes": max_nodes, "max_holes": max_holes,
               "min_tasks": mine.MIN_TASKS, "corpus": sorted(corpus),
               "corpus_nodes": {k: len(v.nodes) for k, v in corpus.items()},
               "eligible": len(props), "ranked": table}
    (OUT / f"proposal_d{depth}_n{max_nodes}_h{max_holes}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True))

    for p in props[:6]:
        print(f"rank {p.rank}: {p.nodes} nodes, {p.holes} holes, saving {p.saving_bits} bits, "
              f"tasks {len(p.tasks)}, occ {p.occurrences}, digest {p.digest[:12]}")

    if not props:
        print("NO ELIGIBLE ABSTRACTION")
        return

    # publish the winner, and the highest-ranked same-arity runner-up as the
    # mined wrong-module control
    top = props[0]
    runner = next((p for p in props[1:] if p.holes == top.holes
                   and mine.truth_table(p.canonical) != mine.truth_table(top.canonical)), None)
    lib = Library(LIB)
    published = {}
    for label, p in (("earned", top), ("earned_runner_up", runner)):
        if p is None:
            continue
        fixture = [{k: Value.of(BOOL, v) for (k, _), v in zip(p.canonical.inputs, bits)}
                   for bits, _ in mine.truth_table(p.canonical)]
        entry = lib.publish(label, p.canonical, Registry(), fixture=fixture,
                            provenance={"rule": "mine.propose", "rank": p.rank,
                                        "corpus_depth": depth, "tasks": list(p.tasks),
                                        "sites": list(p.sites),
                                        "saving_bits": p.saving_bits,
                                        "max_nodes": max_nodes, "max_holes": max_holes})
        published[label] = {"reference": entry.reference, "digest": entry.digest,
                            "operator": entry.operator, "nodes": entry.nodes,
                            "description_bits": entry.description_bits,
                            "execution_cost": entry.execution_cost,
                            "fixture_cases": entry.fixture_cases}
        print("published", label, entry.reference, entry.operator)
    (OUT / "published.json").write_text(json.dumps(published, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
