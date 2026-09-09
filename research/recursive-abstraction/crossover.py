"""How many call sites does a module need before abstraction pays for itself?

Under the repo's own accounting (ARCHITECTURE.md sec. 4) a call is not free:
the module definition is charged once, each call site is charged, and execution
is charged per use. So sharing wins only past a crossover in the number of call
sites. This measures that crossover for MAJ3 (the sub-function of experiment 2),
comparing:

  module form : N call nodes (product(BOOL)) + N projection nodes + library
  flat form   : N inlined copies of or(and(a,b), and(c, xor(a,b)))
"""
from __future__ import annotations

import json
import sys

from tcn.types import BOOL, product
from tcn.operators import Registry
from tcn.graph import Program, Node, Candidate

ONE = product(BOOL)


def maj_module(r):
    """Minimal 4-gate majority, hand-built (no dead scaffold nodes)."""
    return Program(
        (("a", BOOL), ("b", BOOL), ("c", BOOL)),
        (
            Node("t1", BOOL, (Candidate(r.resolve("and", (BOOL, BOOL)), ("a", "b")),), depth=1, selected=0),
            Node("t2", BOOL, (Candidate(r.resolve("xor", (BOOL, BOOL)), ("a", "b")),), depth=1, selected=0),
            Node("t3", BOOL, (Candidate(r.resolve("and", (BOOL, BOOL)), ("c", "t2")),), depth=2, selected=0),
            Node("m", BOOL, (Candidate(r.resolve("or", (BOOL, BOOL)), ("t1", "t3")),), depth=3, selected=0),
        ),
        (("out", "m"),),
    ).validate(r)


def module_form(r, name, n):
    op = r.resolve(name, (BOOL, BOOL, BOOL))
    proj = r.resolve("project", (ONE,), BOOL, {"index": 0})
    nodes = []
    for i in range(n):
        nodes.append(Node(f"call{i}", ONE, (Candidate(op, ("a", "b", "c")),), depth=1, selected=0))
        nodes.append(Node(f"p{i}", BOOL, (Candidate(proj, (f"call{i}",)),), depth=2, selected=0))
    return Program((("a", BOOL), ("b", BOOL), ("c", BOOL)), tuple(nodes),
                   tuple((f"o{i}", f"p{i}") for i in range(n))).validate(r)


def flat_form(r, n):
    nodes = []
    for i in range(n):
        nodes += [
            Node(f"t1_{i}", BOOL, (Candidate(r.resolve("and", (BOOL, BOOL)), ("a", "b")),), depth=1, selected=0),
            Node(f"t2_{i}", BOOL, (Candidate(r.resolve("xor", (BOOL, BOOL)), ("a", "b")),), depth=1, selected=0),
            Node(f"t3_{i}", BOOL, (Candidate(r.resolve("and", (BOOL, BOOL)), ("c", f"t2_{i}")),), depth=2, selected=0),
            Node(f"m_{i}", BOOL, (Candidate(r.resolve("or", (BOOL, BOOL)), (f"t1_{i}", f"t3_{i}")),), depth=3, selected=0),
        ]
    return Program((("a", BOOL), ("b", BOOL), ("c", BOOL)), tuple(nodes),
                   tuple((f"o{i}", f"m_{i}") for i in range(n))).validate(r)


def main():
    r = Registry()
    m = maj_module(r)
    name = r.register_module(m)
    rows = []
    for n in range(1, 13):
        mf = module_form(r, name, n)
        ff = flat_form(r, n)
        rows.append(dict(
            call_sites=n,
            module_bits=mf.description_bits(r), flat_bits=ff.description_bits(r),
            module_bits_no_library=mf.description_bits(),
            module_cost=mf.execution_cost(r), flat_cost=ff.execution_cost(r),
        ))
    crossover_bits = next((x["call_sites"] for x in rows if x["module_bits"] < x["flat_bits"]), None)
    crossover_cost = next((x["call_sites"] for x in rows if x["module_cost"] < x["flat_cost"]), None)
    out = dict(
        module_definition_bits=m.description_bits(),
        module_execution_cost=m.execution_cost(r),
        rows=rows,
        description_bits_crossover_call_sites=crossover_bits,
        execution_cost_crossover_call_sites=crossover_cost,
        note=("execution cost never crosses over: a call is charged exactly the module's internal "
              "cost, and the module form additionally pays one projection node per call site"),
    )
    print(json.dumps(out, indent=2))
    with open(sys.argv[1] if len(sys.argv) > 1 else "crossover.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
