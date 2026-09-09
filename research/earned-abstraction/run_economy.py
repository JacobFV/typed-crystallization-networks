"""Expressive economy: how many scaffold nodes the later task needs per arm.

Search-free. Each route is an explicit straight-line program in that arm's
basis, built as a `tcn.graph.Program` and executed through `tcn` against all 64
rows of the later task. Upper bounds are therefore verified; the flat lower
bound of >= 7 is the retest's gate-elimination result over the full binary
basis B2, which lower-bounds node count in the scaffold basis as well.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from tcn.graph import Program, Node, Candidate
from tcn.types import BOOL, Value

import arms
import later

OUT = Path(__file__).parent / "out"


def build(registry, inputs, gates):
    ports = {k: 0 for k in inputs}
    nodes = []
    for op_name, out, args in gates:
        op = registry.resolve(op_name, tuple(BOOL for _ in args))
        d = max(ports[a] for a in args) + 1
        ports[out] = d
        nodes.append(Node(out, BOOL, (Candidate(op, tuple(args)),), "core", d, 0))
    return Program(tuple((k, BOOL) for k in inputs), tuple(nodes),
                   (("out", nodes[-1].name),)).validate(registry)


def verify(program, registry):
    for bits in itertools.product((False, True), repeat=6):
        out, _ = program.run({k: Value.of(BOOL, v) for k, v in zip(later.INPUTS, bits)},
                             registry=registry)
        if bool(next(iter(out.values())).decoded) != later.composite(bits):
            return False
    return True


def maj_gates(prefix, x, y, z):
    """MAJ3 in the flat basis: 4 nodes."""
    return [("and", prefix + "0", (x, y)), ("or", prefix + "1", (x, y)),
            ("or", prefix + "2", (z, prefix + "0")),
            ("and", prefix + "3", (prefix + "1", prefix + "2"))]


def main():
    rows = []

    # arm 1 -- flat
    r, _ = arms.build("arm1_none")
    gates = maj_gates("p", "a", "b", "c") + maj_gates("q", "d", "e", "f") + \
        [("xor", "y", ("p3", "q3"))]
    p = build(r, later.INPUTS, gates)
    rows.append({"arm": "arm1_none", "nodes": len(p.nodes), "verified": verify(p, r),
                 "description_bits": p.description_bits(r), "execution_cost": p.execution_cost(r),
                 "lower_bound_nodes": 7,
                 "note": "flat lower bound >= 7 by gate elimination over B2 (retest min_program.py)"})

    # arm 2 -- the earned module M(x0,x1,x2) = or(x2, and(x0,x1))
    r, m = arms.build("arm2_earned")
    gates = [("or", "p0", ("a", "b")), (m, "p1", ("a", "b", "c")), ("and", "p2", ("p0", "p1")),
             ("or", "q0", ("d", "e")), (m, "q1", ("d", "e", "f")), ("and", "q2", ("q0", "q1")),
             ("xor", "y", ("p2", "q2"))]
    p = build(r, later.INPUTS, gates)
    rows.append({"arm": "arm2_earned", "nodes": len(p.nodes), "verified": verify(p, r),
                 "description_bits": p.description_bits(r), "execution_cost": p.execution_cost(r),
                 "module": m})

    # arm 3 -- the hand-authored MAJ3
    r, m = arms.build("arm3_authored")
    gates = [(m, "p", ("a", "b", "c")), (m, "q", ("d", "e", "f")), ("xor", "y", ("p", "q"))]
    p = build(r, later.INPUTS, gates)
    rows.append({"arm": "arm3_authored", "nodes": len(p.nodes), "verified": verify(p, r),
                 "description_bits": p.description_bits(r), "execution_cost": p.execution_cost(r),
                 "module": m})

    # arm 4b -- the runner-up M(x0,x1,x2) = and(x2, or(x0,x1))
    r, m = arms.build("arm4b_wrong_mined")
    gates = [("and", "p0", ("a", "b")), ("or", "p1", ("c", "p0")), (m, "p2", ("a", "b", "p1")),
             ("and", "q0", ("d", "e")), ("or", "q1", ("f", "q0")), (m, "q2", ("d", "e", "q1")),
             ("xor", "y", ("p2", "q2"))]
    p = build(r, later.INPUTS, gates)
    rows.append({"arm": "arm4b_wrong_mined", "nodes": len(p.nodes), "verified": verify(p, r),
                 "description_bits": p.description_bits(r), "execution_cost": p.execution_cost(r),
                 "module": m})

    # arm 4 -- the hand-authored distractor. A module arm's scaffold still offers
    # every flat candidate, so its shortest route is the flat one: the module
    # buys nothing. No shorter route is known, and none of <= 3 nodes exists
    # (certified by the tight-scaffold enumeration).
    r, m = arms.build("arm4_wrong_authored")
    gates = maj_gates("p", "a", "b", "c") + maj_gates("q", "d", "e", "f") + \
        [("xor", "y", ("p3", "q3"))]
    p = build(r, later.INPUTS, gates)
    rows.append({"arm": "arm4_wrong_authored", "nodes": len(p.nodes), "verified": verify(p, r),
                 "description_bits": p.description_bits(r), "execution_cost": p.execution_cost(r),
                 "module": m, "module_used": False,
                 "note": "the flat route; the module contributes nothing, and no route of "
                         "<= 3 nodes exists (3-node space exhausted, 0 conforming)"})

    OUT.mkdir(exist_ok=True)
    (OUT / "economy.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
    for row in rows:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
