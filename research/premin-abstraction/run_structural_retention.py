"""Value retention versus structural retention, per corpus, exactly.

*value*      the program holds a node whose value is majority over some
             **ordered 3-subset** of the four inputs;
*structural* the program offers at least one legal fragment (R1: single-exit,
             <= MAX_NODES, <= MAX_HOLES) whose canonical abstraction computes
             majority -- i.e. something the rule could actually rank.

The gap between the two is the interesting quantity: a program can compute the
majority at a node and still offer the rule nothing, because the nodes that
build it are read from outside the fragment and so cannot be replaced by one
call.
"""
from __future__ import annotations

import json
from pathlib import Path

import mine
import mine_multi
import retention
import run_mine_premin as R
import corpora

OUT = Path(__file__).parent / "out"


def maj_valued_nodes(program):
    """Names of nodes whose value is majority over some ordered 3-subset."""
    tables = set(retention.target_tables(retention.maj3).values())
    from tcn.types import BOOL, Value
    import itertools
    from tcn.operators import Registry
    r = Registry()
    names = [n.name for n in program.nodes]
    bits_by_node = {k: 0 for k in names}
    keys = [k for k, _ in program.inputs]
    for bits in itertools.product((False, True), repeat=len(keys)):
        # `itertools.product` counts the first input as the most significant
        # bit; `retention.var_table` counts input j as bit j of the row index,
        # so build the row index the same way.
        idx = sum((1 << j) for j, b in enumerate(bits) if b)
        _, _, trace = program.execute({k: Value.of(BOOL, v) for k, v in zip(keys, bits)},
                                      registry=r)
        for k in names:
            if bool(trace[k].decoded):
                bits_by_node[k] |= 1 << idx
    return {k for k, t in bits_by_node.items() if t in tables}


def main():
    bands = R.load_bands()
    rows = []
    for variant in corpora.VARIANTS:
        corpus, task_of = R.build_variant(variant, bands)
        value_hits, struct_hits = set(), set()
        per_digest = {}
        for entry, program in corpus.items():
            if maj_valued_nodes(program):
                value_hits.add(entry)
            for _root, _S, canon, _holes in mine.fragments(
                    program, mine_multi.MAX_NODES, mine_multi.MAX_HOLES):
                if len(canon.inputs) != 3:
                    continue
                if tuple(v for _, v in mine.truth_table(canon)) != retention.MAJ3_TABLE:
                    continue
                struct_hits.add(entry)
                d = per_digest.setdefault(canon.digest, {"entries": set(), "tasks": set()})
                d["entries"].add(entry)
                d["tasks"].add(task_of[entry])
        rows.append({
            "variant": variant, "entries": len(corpus),
            "entries_with_maj_valued_node": len(value_hits),
            "tasks_with_maj_valued_node": len({task_of[e] for e in value_hits}),
            "entries_offering_a_maj_fragment": len(struct_hits),
            "tasks_offering_a_maj_fragment": len({task_of[e] for e in struct_hits}),
            "distinct_maj_digests": len(per_digest),
            "largest_maj_digest_entries": max((len(v["entries"]) for v in per_digest.values()),
                                              default=0),
            "largest_maj_digest_tasks": max((len(v["tasks"]) for v in per_digest.values()),
                                            default=0),
        })
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
    (OUT / "structural_retention.json").write_text(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
