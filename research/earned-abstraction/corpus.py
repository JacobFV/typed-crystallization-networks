"""The earlier tasks: a small family of solved programs the rule mines.

Nothing here knows about the later task. The family is a set of 4-input
Boolean functions with *latent* shared structure -- each is a 3-of-4 majority
window over some triple of the inputs, combined with the remaining input by
some 2-input gate. That premise is declared openly: library learning is only
meaningful over a task family that shares structure. What is *not* supplied
anywhere is which subprogram realises the shared structure, where it sits in a
solved program, its arity, or that it exists at all. The rule sees only the
solved, pruned typed DAGs.

Each task is solved by the repo's own gradient synthesis on a chain scaffold
built from `registry.resolve` only.
"""
from __future__ import annotations

import itertools
import time

import torch

from tcn.types import BOOL, Value
from tcn.operators import Registry
from tcn.graph import Program, Node, Candidate, Signal
from tcn.learning import SoftProgram, tensor

BOOL_OPS = ("and", "or", "xor")
UNARY_OPS = ("not", "identity")
CORPUS_INPUTS = ("a", "b", "c", "d")


def maj(a, b, c):
    return (int(a) + int(b) + int(c)) >= 2


def _t1(a, b, c, d): return maj(a, b, c) != d
def _t2(a, b, c, d): return maj(a, b, c) and d
def _t3(a, b, c, d): return maj(b, c, d) or a
def _t4(a, b, c, d): return maj(a, c, d) != b
def _t5(a, b, c, d): return maj(a, b, d) or c
def _t6(a, b, c, d): return maj(a, b, c) != maj(b, c, d)


EARLIER_TASKS = (
    ("t1_maj_abc_xor_d", _t1),
    ("t2_maj_abc_and_d", _t2),
    ("t3_maj_bcd_or_a", _t3),
    ("t4_maj_acd_xor_b", _t4),
    ("t5_maj_abd_or_c", _t5),
    ("t6_maj_abc_xor_maj_bcd", _t6),
)


def examples_for(fn, inputs=CORPUS_INPUTS):
    rows = []
    for bits in itertools.product((False, True), repeat=len(inputs)):
        rows.append({"inputs": {k: Value.of(BOOL, v) for k, v in zip(inputs, bits)},
                     "targets": {"out": Value.of(BOOL, bool(fn(*bits)))}})
    return rows


def bool_candidates(r, ports):
    out = []
    names = [k for k, t in ports.items() if t == BOOL]
    for op in BOOL_OPS:
        o = r.resolve(op, (BOOL, BOOL))
        out += [Candidate(o, (p, q)) for p, q in itertools.product(names, repeat=2)]
    for op in UNARY_OPS:
        o = r.resolve(op, (BOOL,))
        out += [Candidate(o, (p,)) for p in names]
    return out


def chain_scaffold(r, inputs=CORPUS_INPUTS, depth=5):
    ports = {k: BOOL for k in inputs}
    nodes = []
    for i in range(depth):
        nodes.append(Node(f"g{i}", BOOL, tuple(bool_candidates(r, ports)), "core", i + 1))
        ports = dict(ports, **{f"g{i}": BOOL})
    program = Program(tuple((k, BOOL) for k in inputs), tuple(nodes),
                      (("out", f"g{depth-1}"),)).validate(r)
    return program, (Signal(f"g{depth-1}", "out", ("core",), BOOL, "bce"),)


def conformant(program, examples, signals, registry):
    for ex in examples:
        try:
            _, _, trace = program.execute(ex["inputs"], registry=registry)
        except (ValueError, TypeError, KeyError, IndexError, OverflowError):
            return False
        for s in signals:
            if trace[s.source].flat() != ex["targets"][s.target].flat():
                return False
    return True


def gradient_solve(program, examples, signals, registry, seed, steps=900, lr=.05, init_noise=.5):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = SoftProgram(program, registry)
    with torch.no_grad():
        for p in model.choices:
            p.add_(torch.randn_like(p) * init_noise)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    inputs = {k: torch.stack([tensor(ex["inputs"][k]) for ex in examples]) for k, _ in program.inputs}
    targets = {s.target: torch.stack([tensor(ex["targets"][s.target]) for ex in examples]) for s in signals}

    def loss_fn():
        _, _, trace = model(inputs, return_trace=True)
        return model.probe_loss(trace, targets, signals)

    t0 = time.perf_counter()
    first = None
    for step in range(steps):
        opt.zero_grad()
        loss = loss_fn() + 0.001 * (step / max(1, steps)) * model.entropy()
        if loss.requires_grad:
            loss.backward()
            opt.step()
        if step % 5 == 0 or step == steps - 1:
            if conformant(model.export(), examples, signals, registry):
                first = step
                break
    wall = time.perf_counter() - t0
    return model, opt, first, wall


def solve_task(fn, seeds=(0, 1, 2, 3, 4, 5, 6, 7), depth=5, inputs=CORPUS_INPUTS, steps=900):
    """Solve one earlier task; return the pruned frozen program and a report."""
    ex = examples_for(fn, inputs)
    attempts = []
    for seed in seeds:
        r = Registry()
        program, signals = chain_scaffold(r, inputs, depth)
        model, opt, first, wall = gradient_solve(program, ex, signals, r, seed, steps=steps)
        if first is None:
            attempts.append({"seed": seed, "conformant": False, "wall_seconds": wall})
            continue
        exported = model.export()
        pruned = exported.pruned().validate(r)
        ok = conformant(pruned, ex, signals, r)
        attempts.append({"seed": seed, "conformant": bool(ok), "first_step": first,
                         "wall_seconds": wall, "raw_nodes": len(exported.nodes),
                         "pruned_nodes": len(pruned.nodes)})
        if ok:
            return pruned, {"solved": True, "seed": seed, "first_step": first,
                            "attempts": attempts, "nodes": len(pruned.nodes),
                            "digest": pruned.digest,
                            "description_bits": pruned.description_bits(),
                            "execution_cost": pruned.execution_cost(r)}
    return None, {"solved": False, "attempts": attempts}
