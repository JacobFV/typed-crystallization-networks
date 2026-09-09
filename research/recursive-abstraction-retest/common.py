"""Shared scaffolds, targets and training loop for the track 5 retest.

Everything runs against unmodified `tcn/` and `generators/`.

The scaffold shape differs from track 5's in one respect, and only because the
F1 fix made the difference possible: a single-output module now resolves to
`BOOL`, so a module call is an ordinary BOOL-valued candidate at an ordinary
BOOL node. Track 5 had to build tuple-typed "call slots" with explicit
projection nodes because `module:<digest>` resolved to `product(BOOL)`. Here
every node is BOOL and the arms differ only in whether that node's candidate
list additionally contains module calls.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import asdict, replace

import torch

from tcn.types import BOOL, Value
from tcn.operators import Registry
from tcn.graph import Program, Node, Candidate, Signal
from tcn.learning import SoftProgram, tensor
from tcn.crystallize import Crystallizer

BOOL_OPS = ("and", "or", "xor")
UNARY_OPS = ("not", "identity")
INPUTS = ("a", "b", "c", "d", "e", "f")
MODULE_ARITY = 3


# --------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------
def maj(a, b, c):
    return (int(a) + int(b) + int(c)) >= 2


def distractor(a, b, c):
    """Truth table 134 -- verified 4 gates, no help toward MAJ3 (min_program.py)."""
    return bool((134 >> (int(a) + 2 * int(b) + 4 * int(c))) & 1)


def composite(bits):
    a, b, c, d, e, f = bits
    return maj(a, b, c) != maj(d, e, f)


def composite_examples():
    rows = []
    for bits in itertools.product((False, True), repeat=6):
        rows.append({"inputs": {k: Value.of(BOOL, v) for k, v in zip(INPUTS, bits)},
                     "targets": {"out": Value.of(BOOL, composite(bits))}})
    return rows


def sub_examples(fn):
    rows = []
    for bits in itertools.product((False, True), repeat=3):
        rows.append({"inputs": {k: Value.of(BOOL, v) for k, v in zip("abc", bits)},
                     "targets": {"out": Value.of(BOOL, fn(*bits))}})
    return rows


# --------------------------------------------------------------------------
# candidate enumeration (every contract comes from registry.resolve)
# --------------------------------------------------------------------------
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
    o = r.resolve(module_name, tuple(BOOL for _ in range(MODULE_ARITY)))
    assert o.output == BOOL, o.output          # the F1 fix; no projection node
    return [Candidate(o, t) for t in itertools.product(pool, repeat=MODULE_ARITY)]


# --------------------------------------------------------------------------
# stage 1: acquire a sub-module on its own 8-row task
# --------------------------------------------------------------------------
def sub_scaffold(r, depth=5):
    ports = {"a": BOOL, "b": BOOL, "c": BOOL}
    nodes = []
    for i in range(depth):
        nodes.append(Node(f"g{i}", BOOL, tuple(bool_candidates(r, ports)), "core", i + 1))
        ports = dict(ports, **{f"g{i}": BOOL})
    return Program((("a", BOOL), ("b", BOOL), ("c", BOOL)), tuple(nodes),
                   (("out", f"g{depth-1}"),)).validate(r), (Signal(f"g{depth-1}", "out", ("core",), BOOL, "bce"),)


# --------------------------------------------------------------------------
# the composite scaffolds
# --------------------------------------------------------------------------
def wide_scaffold(r, module_name=None, workhorse=8):
    """The matched scaffold: `workhorse` BOOL nodes plus an output node.

    Every node sees the six inputs and every earlier node, so the verified
    9-gate flat program fits. In arms B and C every node additionally offers
    `module:<digest>` over the six inputs -- the scaffold makes no guess about
    which node is a "call site".
    """
    inputs = tuple((k, BOOL) for k in INPUTS)
    ports = {k: BOOL for k in INPUTS}
    nodes = []
    names = [f"w{i}" for i in range(workhorse)] + ["y"]
    for depth, name in enumerate(names, start=1):
        cands = bool_candidates(r, ports) + module_candidates(r, module_name, INPUTS)
        nodes.append(Node(name, BOOL, tuple(cands), "core", depth))
        ports = dict(ports, **{name: BOOL})
    return Program(inputs, tuple(nodes), (("out", "y"),)).validate(r)


def tight_scaffold(r, module_name=None):
    """Three nodes: too few for any flat program (the verified minimum is >= 7).

    Only the abstracted route can solve this, so it separates "the abstraction
    does not pay" from "the search cannot find it", and it is small enough for
    `tcn.search.enumerate_fit` to enumerate exhaustively.
    """
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


COMP_SIGNALS = (Signal("y", "out", ("core",), BOOL, "bce"),)


# --------------------------------------------------------------------------
# analysis helpers
# --------------------------------------------------------------------------
def prune(program):
    """Drop nodes unreachable from the outputs.

    `SoftProgram.export()` hardens and keeps the whole scaffold, so an export's
    description bits and execution cost describe the scaffold rather than the
    discovered program (track 5's F4). Every size and cost figure reported for a
    discovered program here is measured after this pass.
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


def module_on_output_path(hardened):
    """Is a module call reachable from the outputs of the discovered program?

    This is track 5's decisive measurement. Selecting a module *somewhere* in
    the argmax means nothing: a module call in a dead branch is not reuse.
    """
    p = prune(hardened)
    return any(n.candidates[n.selected or 0].operator.name.startswith("module:") for n in p.nodes)


def conformant(exported, examples, signals, registry):
    for ex in examples:
        try:
            _, _, trace = exported.execute(ex["inputs"], registry=registry)
        except (ValueError, TypeError, KeyError, IndexError, OverflowError):
            return False
        for s in signals:
            if trace[s.source].flat() != ex["targets"][s.target].flat():
                return False
    return True


# --------------------------------------------------------------------------
# training loop (same objective and schedule as tcn.synthesis.fit, with
# step-resolved instrumentation and search separated from freezing)
# --------------------------------------------------------------------------
def probe_module_mass(model, program):
    """Softmax mass the search puts on module candidates, per node.

    Reported alongside the mass on the single *correct* module binding, so a
    failure to select can be separated into "never considered" and
    "considered and rejected".
    """
    per_node = {}
    for n, logits in zip(program.nodes, model.choices):
        idx = [i for i, c in enumerate(n.candidates) if c.operator.name.startswith("module:")]
        if not idx:
            continue
        q = torch.softmax(logits / model.temperatures[n.name], 0).detach()
        best_correct = 0.
        for i in idx:
            if n.candidates[i].sources in (("a", "b", "c"), ("d", "e", "f")):
                best_correct = max(best_correct, float(q[i]))
        per_node[n.name] = {"module_mass": float(q[list(idx)].sum()),
                            "correct_binding_mass": best_correct,
                            "n_module_candidates": len(idx),
                            "n_candidates": len(n.candidates),
                            "uniform_module_mass": len(idx) / len(n.candidates)}
    return per_node


def search(program, examples, signals, registry, seed, steps=400, lr=0.05,
           eval_every=10, init_noise=0.5, trace_every=0, stop_on_success=True):
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

    first = None
    curve, probes = [], []
    t0 = time.perf_counter()
    last = 0
    for step in range(steps):
        opt.zero_grad()
        loss = loss_fn() + 0.001 * (step / max(1, steps)) * model.entropy()
        if loss.requires_grad:
            loss.backward()
            opt.step()
        last = step
        if trace_every and step % trace_every == 0:
            probes.append({"step": step, "entropy": float(model.entropy().detach()),
                           "loss": float(loss.detach()), "nodes": probe_module_mass(model, program)})
        if step % eval_every == 0 or step == steps - 1:
            ok = conformant(model.export(), examples, signals, registry)
            curve.append({"step": step, "loss": float(loss.detach()), "conformant": ok})
            if ok and first is None:
                first = step
                if stop_on_success:
                    break
    wall = time.perf_counter() - t0
    exported = model.export()
    return model, opt, dict(
        first_conformant_step=first,
        steps_run=last + 1,
        final_loss=float(loss_fn().detach()),
        final_entropy=float(model.entropy().detach()),
        final_conformant=conformant(exported, examples, signals, registry),
        wall_seconds=wall,
        seconds_per_step=wall / max(1, last + 1),
        curve=curve,
        probes=probes,
    ), exported


def crystallize(model, opt, examples, signals, registry, rounds=8, retrain_steps=5, tolerance=0.005):
    inputs = {k: torch.stack([tensor(ex["inputs"][k]) for ex in examples]) for k, _ in model.program.inputs}
    targets = {s.target: torch.stack([tensor(ex["targets"][s.target]) for ex in examples]) for s in signals}

    def loss_fn():
        _, _, trace = model(inputs, return_trace=True)
        return model.probe_loss(trace, targets, signals)

    sched = Crystallizer(model, opt, tolerance=tolerance, entropy_limit=0.9)
    t0 = time.perf_counter()
    sched.run(loss_fn, rounds=rounds, retrain_steps=retrain_steps,
              conformance=lambda ex: conformant(ex, examples, signals, registry))
    reasons = {}
    for e in sched.events:
        reasons[e.reason] = reasons.get(e.reason, 0) + 1
    return dict(attempts=len(sched.events), accepted=sum(1 for e in sched.events if e.accepted),
                reasons=reasons, fully_frozen=len(model.frozen) == len(model.program.nodes),
                wall_seconds=time.perf_counter() - t0)


def acquire_module(fn, seed, attempts=8, steps=800):
    """Learn, crystallize and register a sub-module; acquisition cost is recorded.

    The learned program is pruned before registration. `SoftProgram.export()`
    keeps dead scaffold gates, and a module's dead gates are charged
    transitively at every call site forever (track 5's F4), so registering the
    unpruned export would tax the abstraction for a defect that has nothing to
    do with abstraction. Both sizes are reported.
    """
    ex = sub_examples(fn)
    total = 0
    for k in range(attempts):
        r = Registry()
        prog, signals = sub_scaffold(r)
        model, opt, rep, exported = search(prog, ex, signals, r, seed * 131 + k, steps=steps, eval_every=5)
        total += rep["steps_run"]
        if not rep["final_conformant"]:
            continue
        cry = crystallize(model, opt, ex, signals, r, rounds=6, retrain_steps=5)
        raw = model.export()
        if not conformant(raw, ex, signals, r):
            continue
        pruned = prune(raw).validate(r)
        if not conformant(pruned, ex, signals, r):
            continue
        return pruned, dict(seed=seed, attempts=k + 1, acquisition_steps=total,
                            crystallization=cry, digest=pruned.digest,
                            raw_nodes=len(raw.nodes), pruned_nodes=len(pruned.nodes),
                            raw_description_bits=raw.description_bits(),
                            description_bits=pruned.description_bits(),
                            raw_execution_cost=raw.execution_cost(r),
                            execution_cost=pruned.execution_cost(r))
    return None, dict(seed=seed, attempts=attempts, acquisition_steps=total, digest=None)


def verify_module(module, fn, registry):
    """A registered module must compute exactly the function it claims to."""
    for bits in itertools.product((False, True), repeat=3):
        out, _ = module.run({k: Value.of(BOOL, v) for k, v in zip("abc", bits)}, registry=registry)
        if bool(next(iter(out.values())).decoded) != bool(fn(*bits)):
            return False
    return True


# --------------------------------------------------------------------------
# verified minimal module bodies
# --------------------------------------------------------------------------
# Acquisition and reuse are separate questions. Learning the module is measured
# on its own (`acquire_module`), but the matched arms are supplied with the
# minimum circuit for each sub-function, verified exhaustively by
# `min_program.py` and re-checked against the full truth table at construction.
# That removes two confounds at once: the learned MAJ3 keeps a dead gate (5 live
# nodes where 4 suffice), which would tax abstraction for track 5's F4 rather
# than for abstraction; and the arm-C sub-function is not reliably learnable at
# all by this synthesizer, so arm C could not otherwise be matched to arm B.
MINIMAL_BODIES = {
    # MAJ3(a,b,c) = and(or(a,b), or(c, and(a,b)))
    "maj": (("and", "g0", "a", "b"), ("or", "g1", "a", "b"),
            ("or", "g2", "c", "g0"), ("and", "g3", "g1", "g2")),
    # truth table 134 = xor(xor(a,b), and(c, or(a,b)))
    "distractor": (("or", "g0", "a", "b"), ("xor", "g1", "a", "b"),
                   ("and", "g2", "c", "g0"), ("xor", "g3", "g1", "g2")),
}


def minimal_module(r, kind):
    depth = {"a": 0, "b": 0, "c": 0}
    nodes = []
    for name, out, x, y in MINIMAL_BODIES[kind]:
        op = r.resolve(name, (BOOL, BOOL))
        d = max(depth[x], depth[y]) + 1
        depth[out] = d
        nodes.append(Node(out, BOOL, (Candidate(op, (x, y)),), "core", d, 0))
    p = Program((("a", BOOL), ("b", BOOL), ("c", BOOL)), tuple(nodes),
                (("out", nodes[-1].name),)).validate(r)
    fn = maj if kind == "maj" else distractor
    if not verify_module(p, fn, r):
        raise AssertionError(f"{kind} body does not compute its declared function")
    return p
