"""Track 5's own crossover construction, re-run against the fixed tree.

`costs.py` measures the crossover on the construction this retest actually uses
(N calls on disjoint inputs, combined). Track 5 measured a different one: N
independent copies of MAJ3 on the *same* three inputs, exposed as N outputs,
with a `project` node per call site. The two constructions give different
crossover points, so comparing this retest's numbers against track 5's would
confound the F1 fix with a change of construction.

This reproduces track 5's construction exactly, except that the call node is now
BOOL-typed and the projection node is gone -- which is the whole of the F1 fix.
Both variants are reported, so "what F1 bought" is isolated from "what the
construction assumes".
"""
from __future__ import annotations

import json
from pathlib import Path

from tcn.types import BOOL, product
from tcn.operators import Registry
from tcn.graph import Program, Node, Candidate

import common as C

HERE = Path(__file__).parent
ONE = product(BOOL)


def module_form(r, name, n, with_projection):
    """N call sites. `with_projection` reproduces the pre-F1 shape by hand."""
    op = r.resolve(name, (BOOL, BOOL, BOOL))
    nodes = []
    outs = []
    for i in range(n):
        if with_projection:
            # the pre-fix shape: a call yields product(BOOL) and each site pays a
            # project node. Rebuilt here with an explicit tuple node, since the
            # registry no longer produces a one-field product for a module call.
            nodes.append(Node(f"call{i}", BOOL, (Candidate(op, ("a", "b", "c")),), depth=1, selected=0))
            nodes.append(Node(f"t{i}", ONE, (Candidate(r.resolve("tuple", (BOOL,)), (f"call{i}",)),),
                              depth=2, selected=0))
            nodes.append(Node(f"p{i}", BOOL,
                              (Candidate(r.resolve("project", (ONE,), BOOL, {"index": 0}), (f"t{i}",)),),
                              depth=3, selected=0))
            outs.append(f"p{i}")
        else:
            nodes.append(Node(f"call{i}", BOOL, (Candidate(op, ("a", "b", "c")),), depth=1, selected=0))
            outs.append(f"call{i}")
    return Program((("a", BOOL), ("b", BOOL), ("c", BOOL)), tuple(nodes),
                   tuple((f"o{i}", v) for i, v in enumerate(outs))).validate(r)


def flat_form(r, n):
    """N inlined copies of the minimal MAJ3 body."""
    nodes = []
    for i in range(n):
        nodes += [
            Node(f"t1_{i}", BOOL, (Candidate(r.resolve("and", (BOOL, BOOL)), ("a", "b")),), depth=1, selected=0),
            Node(f"t2_{i}", BOOL, (Candidate(r.resolve("or", (BOOL, BOOL)), ("a", "b")),), depth=1, selected=0),
            Node(f"t3_{i}", BOOL, (Candidate(r.resolve("or", (BOOL, BOOL)), ("c", f"t1_{i}")),), depth=2, selected=0),
            Node(f"m_{i}", BOOL, (Candidate(r.resolve("and", (BOOL, BOOL)), (f"t2_{i}", f"t3_{i}")),), depth=3, selected=0),
        ]
    return Program((("a", BOOL), ("b", BOOL), ("c", BOOL)), tuple(nodes),
                   tuple((f"o{i}", f"m_{i}") for i in range(n))).validate(r)


def sweep(with_projection):
    r = Registry()
    body = C.minimal_module(r, "maj")
    name = r.register_module(body)
    rows = []
    for n in range(1, 13):
        mf = module_form(r, name, n, with_projection)
        ff = flat_form(r, n)
        rows.append(dict(call_sites=n,
                         module_bits=mf.description_bits(r), flat_bits=ff.description_bits(r),
                         module_cost=mf.execution_cost(r), flat_cost=ff.execution_cost(r)))
    return dict(
        module_definition_bits=body.description_bits(),
        module_execution_cost=body.execution_cost(r),
        rows=rows,
        description_bits_crossover=next((x["call_sites"] for x in rows if x["module_bits"] < x["flat_bits"]), None),
        execution_cost_crossover=next((x["call_sites"] for x in rows if x["module_cost"] < x["flat_cost"]), None),
        execution_cost_parity=next((x["call_sites"] for x in rows if x["module_cost"] <= x["flat_cost"]), None),
    )


def main():
    out = {
        "fixed_tree_no_projection": sweep(False),
        "reconstructed_pre_F1_with_projection": sweep(True),
    }
    (HERE / "crossover_track5.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
