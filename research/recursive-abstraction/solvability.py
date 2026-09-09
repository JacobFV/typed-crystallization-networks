"""Sanity gate: both arms' intended solutions must EXIST inside the shared scaffold.

Without this, a 0% success rate could be a scaffold bug rather than a search
result. For each experiment we locate the intended candidate indices by name and
sources, harden the scaffold onto them, and check exact conformance on the full
truth table.
"""
from __future__ import annotations

import json
import sys

from dataclasses import replace

from tcn.operators import Registry

import experiment as E1
import experiment2 as E2


PROGRAMS = {}


def prune(program):
    """Drop nodes unreachable from the outputs.

    tcn exports the whole hardened scaffold, dead nodes included, so
    description_bits/execution_cost of an export measure the scaffold rather
    than the discovered program. This gives the discovered-program figure.
    """
    by_name = {n.name: n for n in program.nodes}
    keep, stack = set(), [v for _, v in program.outputs] + [u for _, _, u in program.state]
    while stack:
        k = stack.pop()
        if k in by_name and k not in keep:
            keep.add(k)
            n = by_name[k]
            stack += list(n.candidates[n.selected or 0].sources)
    return replace(program, nodes=tuple(n for n in program.nodes if n.name in keep))


def metrics(hard, registry):
    p = prune(hard)
    return dict(
        nodes=len(hard.nodes), description_bits=hard.description_bits(registry),
        execution_cost=hard.execution_cost(registry),
        pruned_nodes=len(p.nodes), pruned_description_bits=p.description_bits(registry),
        pruned_description_bits_no_library=p.description_bits(),
        pruned_execution_cost=p.execution_cost(registry),
    )


def find(program, node_name, op_predicate):
    node = next(n for n in program.nodes if n.name == node_name)
    for i, c in enumerate(node.candidates):
        if op_predicate(c):
            return i
    raise LookupError(f"{node_name}: no candidate matches")


def check(program, selections, examples, signals, registry):
    hard = program.harden(selections).validate(registry)
    return E1.conformant(hard, examples, signals, registry), hard


def e1_solutions():
    ex = E1.composite_examples()
    res = {}

    # ---- arm A: flat 5-gate full adder ----------------------------------
    r = Registry()
    p = E1.composite_scaffold(r)
    sel = {
        "call1": find(p, "call1", lambda c: c.operator.name == "tuple" and c.sources == ("a", "a")),
        "p1s": find(p, "p1s", lambda c: c.operator.name == "xor" and c.sources == ("a", "b")),
        "p1c": find(p, "p1c", lambda c: c.operator.name == "and" and c.sources == ("a", "b")),
        "call2": find(p, "call2", lambda c: c.operator.name == "tuple" and c.sources == ("a", "a")),
        "p2s": find(p, "p2s", lambda c: c.operator.name == "xor" and c.sources == ("p1s", "cin")),
        "p2c": find(p, "p2c", lambda c: c.operator.name == "and" and c.sources == ("p1s", "cin")),
        "w1": find(p, "w1", lambda c: c.operator.name == "or" and c.sources == ("p1c", "p2c")),
        "w2": find(p, "w2", lambda c: c.operator.name == "identity" and c.sources == ("a",)),
        "sum_out": find(p, "sum_out", lambda c: c.operator.name == "identity" and c.sources == ("p2s",)),
        "carry_out": find(p, "carry_out", lambda c: c.operator.name == "identity" and c.sources == ("w1",)),
    }
    ok, hard = check(p, sel, ex, E1.FA_SIGNALS, r)
    res["e1_armA_flat"] = dict(conformant=ok, constrained_choices=7, **metrics(hard, r))
    PROGRAMS["e1_armA_flat"] = (hard, r)

    # ---- arm B: two half-adder calls -------------------------------------
    module, mreg, _ = E1.build_module(1000)
    r2 = Registry()
    name = r2.register_module(module)
    p2 = E1.composite_scaffold(r2, name)
    sel2 = {
        "call1": find(p2, "call1", lambda c: c.operator.name == name and c.sources == ("a", "b")),
        "p1s": find(p2, "p1s", lambda c: c.operator.name == "project" and dict(c.operator.parameters)["index"] == 0),
        "p1c": find(p2, "p1c", lambda c: c.operator.name == "project" and dict(c.operator.parameters)["index"] == 1),
        "call2": find(p2, "call2", lambda c: c.operator.name == name and c.sources == ("p1s", "cin")),
        "p2s": find(p2, "p2s", lambda c: c.operator.name == "project" and dict(c.operator.parameters)["index"] == 0),
        "p2c": find(p2, "p2c", lambda c: c.operator.name == "project" and dict(c.operator.parameters)["index"] == 1),
        "w1": find(p2, "w1", lambda c: c.operator.name == "or" and c.sources == ("p1c", "p2c")),
        "w2": find(p2, "w2", lambda c: c.operator.name == "identity" and c.sources == ("a",)),
        "sum_out": find(p2, "sum_out", lambda c: c.operator.name == "identity" and c.sources == ("p2s",)),
        "carry_out": find(p2, "carry_out", lambda c: c.operator.name == "identity" and c.sources == ("w1",)),
    }
    ok2, hard2 = check(p2, sel2, ex, E1.FA_SIGNALS, r2)
    res["e1_armB_module"] = dict(conformant=ok2, constrained_choices=8,
                                module_bits=module.description_bits(),
                                module_cost=module.execution_cost(r2), **metrics(hard2, r2))
    PROGRAMS["e1_armB_module"] = (hard2, r2)
    return res


def e2_solutions():
    ex = E2.composite_examples()
    res = {}
    # ---- arm A: 9-gate flat ---------------------------------------------
    r = Registry()
    p = E2.composite_scaffold(r)
    # maj(a,b,c) = or(and(a,b), and(c, xor(a,b))); maj(b,c,d) likewise
    sel = {}
    sel["call1"] = find(p, "call1", lambda c: c.operator.name == "tuple")
    sel["p1"] = find(p, "p1", lambda c: c.operator.name == "and" and c.sources == ("a", "b"))
    sel["call2"] = find(p, "call2", lambda c: c.operator.name == "tuple")
    sel["p2"] = find(p, "p2", lambda c: c.operator.name == "xor" and c.sources == ("a", "b"))
    plan = [
        ("w0", "and", ("c", "p2")),      # c & (a^b)
        ("w1", "or", ("p1", "w0")),      # maj(a,b,c)
        ("w2", "and", ("b", "c")),
        ("w3", "xor", ("b", "c")),
        ("w4", "and", ("d", "w3")),
        ("w5", "or", ("w2", "w4")),      # maj(b,c,d)
        ("w6", "identity", ("w1",)),     # carry maj1 into the bounded window
        ("w7", "xor", ("w6", "w5")),     # the answer
    ]
    for name, op, src in plan:
        sel[name] = find(p, name, lambda c, op=op, src=src: c.operator.name == op and c.sources == src)
    sel["y"] = find(p, "y", lambda c: c.operator.name == "identity" and c.sources == ("w7",))
    ok, hard = check(p, sel, ex, E2.COMP_SIGNALS, r)
    res["e2_armA_flat"] = dict(conformant=ok, constrained_choices=11, **metrics(hard, r))
    PROGRAMS["e2_armA_flat"] = (hard, r)

    # ---- arm B: two MAJ3 calls -------------------------------------------
    module, mrep = E2.build_maj_module(0)
    if module is None:
        res["e2_armB_module"] = dict(conformant=None, note="module acquisition failed")
        return res
    r2 = Registry()
    name = r2.register_module(module)
    p2 = E2.composite_scaffold(r2, name)
    sel2 = {
        "call1": find(p2, "call1", lambda c: c.operator.name == name and c.sources == ("a", "b", "c")),
        "p1": find(p2, "p1", lambda c: c.operator.name == "project"),
        "call2": find(p2, "call2", lambda c: c.operator.name == name and c.sources == ("b", "c", "d")),
        "p2": find(p2, "p2", lambda c: c.operator.name == "project"),
    }
    # carry p1/p2 forward through the workhorse chain to reach y's bounded pool
    sel2["w0"] = find(p2, "w0", lambda c: c.operator.name == "identity" and c.sources == ("p1",))
    sel2["w1"] = find(p2, "w1", lambda c: c.operator.name == "identity" and c.sources == ("p2",))
    for i in range(2, 8):
        sel2[f"w{i}"] = find(p2, f"w{i}", lambda c: c.operator.name == "identity" and c.sources == ("a",))
    sel2["y"] = find(p2, "y", lambda c: c.operator.name == "xor" and set(c.sources) == {"p1", "p2"})
    ok2, hard2 = check(p2, sel2, ex, E2.COMP_SIGNALS, r2)
    res["e2_armB_module"] = dict(conformant=ok2, constrained_choices=5,
                                 module_bits=module.description_bits(),
                                 module_cost=module.execution_cost(r2), **metrics(hard2, r2))
    PROGRAMS["e2_armB_module"] = (hard2, r2)
    return res


if __name__ == "__main__":
    out = {}
    out.update(e1_solutions())
    out.update(e2_solutions())
    print(json.dumps(out, indent=2))
    with open(sys.argv[1] if len(sys.argv) > 1 else "solvability.json", "w") as f:
        json.dump(out, f, indent=2)
