"""Track 5, experiment 2: a DEEP reusable sub-function.

Experiment 1 (experiment.py) reuses a 2-gate half adder, which saves the flat
arm essentially no nodes. Experiment 2 raises the stake:

  sub-function  MAJ3(a,b,c) = 2-of-3 majority (the carry function)
                minimum 4 two-input gates: or(and(a,b), and(c, xor(a,b)))
                NOT available as a primitive: `truth_k` covers every 2-input
                Boolean function, but no 3-input one.
  composite     H(a,b,c,d) = MAJ3(a,b,c) xor MAJ3(b,c,d)   (sliding window)
                flat minimum: 4 + 4 + 1 = 9 two-input gates
                with the module: 2 calls + 2 projections + 1 xor

So abstraction should save ~7 learned gate choices. 4 Boolean inputs => the
complete 16-row truth table is both training set and conformance set.

Both arms share one scaffold with identical nodes, depths, bounded predecessor
pools and Boolean vocabulary. The two tuple-valued call slots offer
`tuple(p)` in both arms and additionally `module:MAJ3(p,q,r)` in arm B.
(A one-output module resolves to product(BOOL), never BOOL -- see
accounting_check.py M1 -- hence the explicit projection nodes.)
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from pathlib import Path

import torch

from tcn.types import BOOL, Value, product
from tcn.operators import Registry
from tcn.graph import Program, Node, Candidate, Signal

import experiment as E

ONE = product(BOOL)


def maj(a, b, c):
    return (int(a) + int(b) + int(c)) >= 2


# ----------------------------- stage 1: MAJ3 ------------------------------
def maj_scaffold(r, depth=5):
    ports = {"a": BOOL, "b": BOOL, "c": BOOL}
    nodes = []
    for i in range(depth):
        nodes.append(Node(f"g{i}", BOOL, E.bool_candidates(r, ports), "core", i + 1))
        ports = dict(ports, **{f"g{i}": BOOL})
    return Program(
        (("a", BOOL), ("b", BOOL), ("c", BOOL)), tuple(nodes), (("out", f"g{depth-1}"),)
    ).validate(r)


MAJ_SIGNALS = (Signal("g4", "out", ("core",), BOOL, "bce"),)


def maj_examples():
    rows = []
    for a in (False, True):
        for b in (False, True):
            for c in (False, True):
                rows.append(
                    {
                        "inputs": {"a": Value.of(BOOL, a), "b": Value.of(BOOL, b), "c": Value.of(BOOL, c)},
                        "targets": {"out": Value.of(BOOL, maj(a, b, c))},
                    }
                )
    return rows


def build_maj_module(seed, attempts=6, steps=600):
    """Acquire the MAJ3 module; retries count toward arm B's acquisition cost."""
    ex = maj_examples()
    total_steps = 0
    tries = 0
    for k in range(attempts):
        tries += 1
        r = Registry()
        prog = maj_scaffold(r)
        model, opt, rep, exported = E.search(prog, ex, MAJ_SIGNALS, r, seed * 100 + k,
                                             steps=steps, eval_every=5)
        total_steps += rep["steps_run"]
        if rep["final_conformant"]:
            cry = E.crystallize(model, opt, ex, MAJ_SIGNALS, r, rounds=6, retrain_steps=5)
            module = model.export()
            if E.conformant(module, ex, MAJ_SIGNALS, r):
                return module, dict(
                    seed=seed, attempts=tries, acquisition_steps=total_steps,
                    crystallization=cry, digest=module.digest,
                    description_bits=module.description_bits(),
                    execution_cost=module.execution_cost(r),
                )
    return None, dict(seed=seed, attempts=tries, acquisition_steps=total_steps, digest=None)


# --------------------------- composite scaffold ---------------------------
def call_slot_candidates(r, pool, module_name):
    tup = r.resolve("tuple", (BOOL,))
    out = [Candidate(tup, (p,)) for p in pool]
    if module_name is not None:
        mop = r.resolve(module_name, (BOOL, BOOL, BOOL))
        assert mop.output == ONE, mop.output
        out += [Candidate(mop, t) for t in itertools.product(pool, repeat=3)]
    return tuple(out)


INPUTS = ("a", "b", "c", "d")


def composite_scaffold(r, module_name=None, workhorse=8, pool_window=5):
    inputs = tuple((k, BOOL) for k in INPUTS)
    base = {k: BOOL for k in INPUTS}
    nodes = []

    # call slots draw from a bounded predecessor pool: the four inputs
    nodes.append(Node("call1", ONE, call_slot_candidates(r, INPUTS, module_name), "core", 1))
    proj1 = E.projections(r, "call1", ONE)
    nodes.append(Node("p1", BOOL, E.bool_candidates(r, base, proj1), "core", 2))

    nodes.append(Node("call2", ONE, call_slot_candidates(r, INPUTS, module_name), "core", 3))
    proj2 = E.projections(r, "call2", ONE)
    nodes.append(Node("p2", BOOL, E.bool_candidates(r, dict(base, p1=BOOL), proj2), "core", 4))

    recent = ["p1", "p2"]
    depth = 5
    for i in range(workhorse):
        ports = dict(base, **{k: BOOL for k in recent[-pool_window:]})
        nodes.append(Node(f"w{i}", BOOL, E.bool_candidates(r, ports), "core", depth))
        recent.append(f"w{i}")
        depth += 1
    ports = dict(base, p1=BOOL, p2=BOOL, **{k: BOOL for k in recent[-pool_window:]})
    nodes.append(Node("y", BOOL, E.bool_candidates(r, ports), "core", depth))
    return Program(inputs, tuple(nodes), (("out", "y"),)).validate(r)


COMP_SIGNALS = (Signal("y", "out", ("core",), BOOL, "bce"),)


def composite_examples():
    rows = []
    for bits in itertools.product((False, True), repeat=4):
        a, b, c, d = bits
        rows.append(
            {
                "inputs": {k: Value.of(BOOL, v) for k, v in zip(INPUTS, bits)},
                "targets": {"out": Value.of(BOOL, maj(a, b, c) != maj(b, c, d))},
            }
        )
    return rows


def run_arm(arm, seed, steps, module):
    r = Registry()
    name = r.register_module(module) if arm == "B" else None
    prog = composite_scaffold(r, name)
    ex = composite_examples()
    model, opt, rep, exported = E.search(prog, ex, COMP_SIGNALS, r, seed, steps=steps, eval_every=10)
    rep.update(
        arm=arm, seed=seed,
        scaffold_nodes=len(prog.nodes),
        scaffold_candidates=sum(len(n.candidates) for n in prog.nodes),
        final_description_bits=exported.description_bits(r),
        final_description_bits_no_registry=exported.description_bits(),
        final_execution_cost=exported.execution_cost(r),
        used_module=any(
            prog.nodes[i].candidates[model.selections()[prog.nodes[i].name]].operator.name.startswith("module:")
            for i in range(len(prog.nodes))
        ),
        selections={
            n.name: {
                "operator": n.candidates[model.selections()[n.name]].operator.name[:14],
                "sources": list(n.candidates[model.selections()[n.name]].sources),
            }
            for n in prog.nodes
        },
    )
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--out", default="e2_results.json")
    args = ap.parse_args()
    out = {"config": vars(args), "module_stage": [], "runs": []}

    modules = {}
    for seed in range(args.seeds):
        mod, mrep = build_maj_module(seed)
        modules[seed] = mod
        out["module_stage"].append(mrep)
        print(f"[maj] seed={seed} ok={mod is not None} attempts={mrep['attempts']} "
              f"steps={mrep['acquisition_steps']} bits={mrep.get('description_bits')} "
              f"cost={mrep.get('execution_cost')}", flush=True)

    for seed in range(args.seeds):
        for arm in ("A", "B"):
            if arm == "B" and modules[seed] is None:
                continue
            t = time.perf_counter()
            rep = run_arm(arm, seed, args.steps, modules[seed])
            rep["total_wall_seconds"] = time.perf_counter() - t
            out["runs"].append(rep)
            print(f"[{arm}] seed={seed} conf={rep['final_conformant']} "
                  f"first={rep['first_conformant_step']} loss={rep['final_loss']:.4f} "
                  f"cands={rep['scaffold_candidates']} bits={rep['final_description_bits']} "
                  f"cost={rep['final_execution_cost']} used_module={rep['used_module']} "
                  f"t={rep['total_wall_seconds']:.0f}s", flush=True)
            Path(args.out).write_text(json.dumps(out, indent=2, default=str))

    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    for arm in ("A", "B"):
        runs = [r for r in out["runs"] if r["arm"] == arm]
        if not runs:
            continue
        succ = [r for r in runs if r["final_conformant"]]
        print(f"\narm {arm}: success {len(succ)}/{len(runs)}")
        if succ:
            st = [r["first_conformant_step"] for r in succ]
            print(f"  steps median={statistics.median(st)} min={min(st)} max={max(st)}")
        print(f"  final loss median={statistics.median([r['final_loss'] for r in runs]):.4f}")
        print(f"  bits median={statistics.median([r['final_description_bits'] for r in runs])}")
        print(f"  cost median={statistics.median([r['final_execution_cost'] for r in runs])}")
        print(f"  sec/step median={statistics.median([r['seconds_per_step'] for r in runs]):.3f}")


if __name__ == "__main__":
    main()
