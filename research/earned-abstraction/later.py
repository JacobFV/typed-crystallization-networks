"""The later task and the four arms.

The later task is fixed before the rule runs and the rule never sees it:
`maj(a,b,c) xor maj(d,e,f)` over six Boolean inputs, on the scaffolds of
`research/recursive-abstraction-retest` -- the current state of the art for
this question, so arm 3 is literally the module the existing demonstration
hand-authored, in the same scaffold, with the same candidate construction.

Scaffolds are rebuilt here rather than imported only because the retest fixes
`MODULE_ARITY = 3`; an earned module's arity is whatever the rule proposed, so
the module candidate block has to follow the module's own signature. With a
3-ary module these functions produce candidate counts identical to the
retest's (120/120/16 flat, 336/336/24 with a module).
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
INPUTS = ("a", "b", "c", "d", "e", "f")


def maj(a, b, c):
    return (int(a) + int(b) + int(c)) >= 2


def composite(bits):
    a, b, c, d, e, f = bits
    return maj(a, b, c) != maj(d, e, f)


def later_examples():
    rows = []
    for bits in itertools.product((False, True), repeat=6):
        rows.append({"inputs": {k: Value.of(BOOL, v) for k, v in zip(INPUTS, bits)},
                     "targets": {"out": Value.of(BOOL, composite(bits))}})
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


def module_candidates(r, module_name, pool):
    if module_name is None:
        return []
    m = r.modules[module_name]
    arity = len(m.inputs)
    o = r.resolve(module_name, tuple(t for _, t in m.inputs))
    if o.output != BOOL:
        raise TypeError("this scaffold offers BOOL-valued module calls only")
    return [Candidate(o, t) for t in itertools.product(pool, repeat=arity)]


def tight_scaffold(r, module_name=None):
    """Three nodes. The flat minimum for the target is proved >= 7, so arm 1
    cannot solve it at all; the space is small enough to exhaust."""
    inputs = tuple((k, BOOL) for k in INPUTS)
    base = {k: BOOL for k in INPUTS}
    nodes = [
        Node("n1", BOOL, tuple(bool_candidates(r, base) + module_candidates(r, module_name, INPUTS)), "core", 1),
        Node("n2", BOOL, tuple(bool_candidates(r, base) + module_candidates(r, module_name, INPUTS)), "core", 1),
    ]
    top = {"n1": BOOL, "n2": BOOL}
    nodes.append(Node("y", BOOL, tuple(bool_candidates(r, top) +
                                       module_candidates(r, module_name, ("n1", "n2"))), "core", 2))
    return Program(inputs, tuple(nodes), (("out", "y"),)).validate(r)


def wide_scaffold(r, module_name=None, workhorse=8):
    """Nine nodes: wide enough for the verified 9-gate flat program, so both
    routes fit and the comparison is not degenerate. Far too large to exhaust."""
    inputs = tuple((k, BOOL) for k in INPUTS)
    ports = {k: BOOL for k in INPUTS}
    nodes = []
    names = [f"w{i}" for i in range(workhorse)] + ["y"]
    for depth, name in enumerate(names, start=1):
        cands = bool_candidates(r, ports) + module_candidates(r, module_name, INPUTS)
        nodes.append(Node(name, BOOL, tuple(cands), "core", depth))
        ports = dict(ports, **{name: BOOL})
    return Program(inputs, tuple(nodes), (("out", "y"),)).validate(r)


SIGNALS = (Signal("y", "out", ("core",), BOOL, "bce"),)


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


def accuracy(program, examples, signals, registry):
    hit = 0
    for ex in examples:
        try:
            _, _, trace = program.execute(ex["inputs"], registry=registry)
        except (ValueError, TypeError, KeyError, IndexError, OverflowError):
            continue
        if all(trace[s.source].flat() == ex["targets"][s.target].flat() for s in signals):
            hit += 1
    return hit / len(examples)


def module_on_output_path(hardened):
    p = hardened.pruned()
    return any(n.candidates[n.selected or 0].operator.name.startswith("module:") for n in p.nodes)


def gradient_search(program, examples, signals, registry, seed, steps=400, lr=.05,
                    eval_every=10, init_noise=.5):
    """The retest's loop, unchanged: same objective, schedule and stopping rule."""
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
    first, last = None, 0
    for step in range(steps):
        opt.zero_grad()
        loss = loss_fn() + 0.001 * (step / max(1, steps)) * model.entropy()
        if loss.requires_grad:
            loss.backward()
            opt.step()
        last = step
        if step % eval_every == 0 or step == steps - 1:
            if conformant(model.export(), examples, signals, registry):
                first = step
                break
    wall = time.perf_counter() - t0
    exported = model.export()
    return {"seed": seed, "first_conformant_step": first, "steps_run": last + 1,
            "conformant": conformant(exported, examples, signals, registry),
            "accuracy": accuracy(exported, examples, signals, registry),
            "module_on_output_path": module_on_output_path(exported),
            "wall_seconds": wall, "seconds_per_step": wall / max(1, last + 1)}


# --------------------------------------------------------------------- modules

MINIMAL_BODIES = {
    # MAJ3(a,b,c) = and(or(a,b), or(c, and(a,b))) -- the existing demonstration's
    # hand-authored module, verified minimal by the retest's `min_program.py`.
    "maj": (("and", "g0", "a", "b"), ("or", "g1", "a", "b"),
            ("or", "g2", "c", "g0"), ("and", "g3", "g1", "g2")),
    # truth table 134 = xor(xor(a,b), and(c, or(a,b))) -- same size, same arity,
    # verified by the retest to shorten MAJ3 by nothing.
    "distractor": (("or", "g0", "a", "b"), ("xor", "g1", "a", "b"),
                   ("and", "g2", "c", "g0"), ("xor", "g3", "g1", "g2")),
}


def hand_authored(r, kind):
    depth = {"a": 0, "b": 0, "c": 0}
    nodes = []
    for name, out, x, y in MINIMAL_BODIES[kind]:
        op = r.resolve(name, (BOOL, BOOL))
        d = max(depth[x], depth[y]) + 1
        depth[out] = d
        nodes.append(Node(out, BOOL, (Candidate(op, (x, y)),), "core", d, 0))
    return Program((("a", BOOL), ("b", BOOL), ("c", BOOL)), tuple(nodes),
                   (("out", nodes[-1].name),)).validate(r)
