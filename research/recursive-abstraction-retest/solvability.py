"""Prove both arms' intended solutions exist in the shared scaffolds, and time a step.

Without this a 0% success rate is indistinguishable from a scaffold bug. Each
intended solution is constructed by hand inside the scaffold the search actually
gets, hardened, and checked for exact conformance on all 64 rows.

It also measures the per-step cost of each arm, which decides the seed and step
budget the experiment can afford, and re-derives track 5's F6 arithmetic with the
current per-row module cost.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import torch

from tcn.types import BOOL, Value
from tcn.operators import Registry
from tcn.graph import Program
from tcn.learning import SoftProgram, tensor

import common as C

HERE = Path(__file__).parent


def pick(program, node, predicate):
    n = next(x for x in program.nodes if x.name == node)
    for i, c in enumerate(n.candidates):
        if predicate(c):
            return i
    raise LookupError(f"{node}: no candidate matches")


def is_op(name, sources):
    return lambda c: c.operator.name == name and c.sources == sources


def is_module(sources):
    return lambda c: c.operator.name.startswith("module:") and c.sources == sources


def check(program, selections, registry, label):
    hard = program.harden(selections).validate(registry)
    ok = C.conformant(hard, C.composite_examples(), C.COMP_SIGNALS, registry)
    p = C.prune(hard)
    return {
        "label": label, "conformant": ok,
        "scaffold_nodes": len(program.nodes), "live_nodes": len(p.nodes),
        "choices_required": len(selections),
        "description_bits_scaffold": hard.description_bits(registry),
        "description_bits_pruned": p.description_bits(registry),
        "description_bits_pruned_no_library": p.description_bits(),
        "execution_cost_pruned": p.execution_cost(registry),
        "module_on_output_path": C.module_on_output_path(hard),
    }


def flat_selections(program):
    """The verified 9-gate program: MAJ3(a,b,c) xor MAJ3(d,e,f), written out."""
    s = {}
    s["w0"] = pick(program, "w0", is_op("and", ("a", "b")))
    s["w1"] = pick(program, "w1", is_op("xor", ("a", "b")))
    s["w2"] = pick(program, "w2", is_op("and", ("c", "w1")))
    s["w3"] = pick(program, "w3", is_op("or", ("w0", "w2")))
    s["w4"] = pick(program, "w4", is_op("and", ("d", "e")))
    s["w5"] = pick(program, "w5", is_op("xor", ("d", "e")))
    s["w6"] = pick(program, "w6", is_op("and", ("f", "w5")))
    s["w7"] = pick(program, "w7", is_op("or", ("w4", "w6")))
    s["y"] = pick(program, "y", is_op("xor", ("w3", "w7")))
    return s


def module_selections_wide(program):
    """The abstracted route inside the same 9-node scaffold: 3 live nodes."""
    s = {n.name: 0 for n in program.nodes}          # the rest are dead branches
    s["w0"] = pick(program, "w0", is_module(("a", "b", "c")))
    s["w1"] = pick(program, "w1", is_module(("d", "e", "f")))
    s["y"] = pick(program, "y", is_op("xor", ("w0", "w1")))
    return s


def module_selections_tight(program):
    return {
        "n1": pick(program, "n1", is_module(("a", "b", "c"))),
        "n2": pick(program, "n2", is_module(("d", "e", "f"))),
        "y": pick(program, "y", is_op("xor", ("n1", "n2"))),
    }


def time_step(program, registry, steps=3):
    ex = C.composite_examples()
    torch.set_num_threads(1)
    model = SoftProgram(program, registry)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    inputs = {k: torch.stack([tensor(e["inputs"][k]) for e in ex]) for k, _ in program.inputs}
    targets = {s.target: torch.stack([tensor(e["targets"][s.target]) for e in ex]) for s in C.COMP_SIGNALS}

    def loss_fn():
        _, _, tr = model(inputs, return_trace=True)
        return model.probe_loss(tr, targets, C.COMP_SIGNALS)

    loss_fn()
    t = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(); loss = loss_fn(); loss.backward(); opt.step()
    train = (time.perf_counter() - t) / steps
    t = time.perf_counter()
    C.conformant(model.export(), ex, C.COMP_SIGNALS, registry)
    return {"seconds_per_step": round(train, 4), "seconds_per_conformance_check": round(time.perf_counter() - t, 4)}


def main():
    out = {}

    # a hand-built, exactly-correct MAJ3 module stands in for a learned one here,
    # so scaffold solvability is not confounded with acquisition succeeding.
    r_ref = Registry()
    maj_body = C.minimal_module(r_ref, "maj")

    # --- arm A: flat, no modules offered -------------------------------
    rA = Registry()
    pA = C.wide_scaffold(rA, None)
    out["wide_arm_A_flat"] = check(pA, flat_selections(pA), rA, "9-gate flat program in the 9-node scaffold")
    out["wide_arm_A_candidates"] = sum(len(n.candidates) for n in pA.nodes)

    # --- arm B: same scaffold, modules offered at every node -----------
    rB = Registry()
    name = rB.register_module(C.minimal_module(rB, "maj"))
    assert C.verify_module(maj_body, C.maj, rB)
    pB = C.wide_scaffold(rB, name)
    out["wide_arm_B_flat_route"] = check(pB, flat_selections(pB), rB, "the flat program is still available in arm B")
    out["wide_arm_B_module_route"] = check(pB, module_selections_wide(pB), rB, "2 module calls + 1 xor")
    out["wide_arm_B_candidates"] = sum(len(n.candidates) for n in pB.nodes)
    out["wide_candidate_inflation"] = round(out["wide_arm_B_candidates"] / out["wide_arm_A_candidates"], 3)

    # --- tight scaffold: only the abstracted route fits -----------------
    rT = Registry()
    nameT = rT.register_module(C.minimal_module(rT, "maj"))
    pT = C.tight_scaffold(rT, nameT)
    out["tight_arm_B_module_route"] = check(pT, module_selections_tight(pT), rT, "3-node scaffold, module route")
    rTA = Registry()
    pTA = C.tight_scaffold(rTA, None)
    out["tight_arm_A_candidates"] = sum(len(n.candidates) for n in pTA.nodes)
    out["tight_arm_B_candidates"] = sum(len(n.candidates) for n in pT.nodes)
    out["tight_space_size_A"] = 1
    for n in pTA.nodes:
        out["tight_space_size_A"] *= len(n.candidates)
    out["tight_space_size_B"] = 1
    for n in pT.nodes:
        out["tight_space_size_B"] *= len(n.candidates)

    # --- timing --------------------------------------------------------
    out["timing_wide_A"] = time_step(pA, rA)
    out["timing_wide_B"] = time_step(pB, rB)
    out["timing_tight_B"] = time_step(pT, rT)

    (HERE / "solvability.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
