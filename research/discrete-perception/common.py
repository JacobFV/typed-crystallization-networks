"""Shared machinery for the discrete-perception track.

The architecture under test, in one sentence: **search the per-position module
discretely with `tcn.search.enumerate_fit`, then apply it at every position with
the three-node `insert`/`pair`/`map` pattern.**

Nothing under `tcn/` or `generators/` is modified.  Two things are copied in
here rather than imported, and both are named:

* `positional_caller` is the three-node pattern from the (unmerged)
  `positional-reuse` branch's `tcn.scaffold.positional_scaffold`, reproduced
  locally so this track does not depend on that branch.
* `local_fit` is `tcn.synthesis.fit`'s control flow plus an `init_noise`
  argument.  `SoftProgram` zero-initialises every choice logit, so
  `torch.manual_seed` does not vary synthesis at all; without explicit noise a
  "12 seed" number is one outcome repeated twelve times.  Every gradient number
  in this track either says `init_noise` or is reported as a single outcome.

Rules honoured (AGENTS.md): only `record.actor_view().observations` reaches a
program input; `latent_states` and `probes` are used exclusively to build
supervision targets.
"""
from __future__ import annotations

import itertools
import json
import math
import pathlib
import statistics
import time

import torch

from tcn.crystallize import Crystallizer
from tcn.generation import Action, Host, SCALAR, Value
from tcn.graph import Candidate, Node, Program, Signal
from tcn.learning import SoftProgram, tensor as _tensor
from tcn.operators import Registry
from tcn.search import enumerate_fit, evaluate, space_size
from tcn.types import BOOL, integer, product, setof

VEC3 = product(SCALAR, SCALAR, SCALAR)
BYTE = integer(8, signed=False, role="byte")
IDX = integer(16, signed=False)
BACKGROUND = (24, 30, 43)

OUT = pathlib.Path(__file__).resolve().parent / "out"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def approach(delta=(-2.4, -2.4, -4.2)):
    """One `camera` action from the generator's declared action schema.

    At the shipped camera an object subtends ~0.07*R pixels and 98% of every
    image is background at every resolution (perception-ladder section 4.2), so
    any accuracy figure there is a report on a constant.  This is a declared
    action, not a code change.
    """
    return Action("camera", arguments=(("delta", Value.of(VEC3, delta)),))


def episode(seed, resolution, objects=6, split="train", delta=(-2.4, -2.4, -4.2)):
    """Raw observation plus the privileged probes, kept physically separate."""
    h = Host.create("geometry", seed=seed, split=split,
                    configuration={"resolution": resolution, "objects": objects, "horizon": 4})
    rec = h.step([approach(delta)])
    view = rec.actor_view()
    assert set(view.observations) == {"pixels"}, view.observations
    pixels = view.observations["pixels"].decoded[3]          # the raw byte tuple
    probes = {k: rec.probes[k].decoded for k in ("depth", "object_ids", "normals")}
    return pixels, probes


def bytes_type(resolution):
    return product(*(BYTE for _ in range(3 * resolution * resolution)))


def record_type(resolution):
    return product(IDX, bytes_type(resolution))


def pixel_starts(resolution):
    return tuple(3 * i for i in range(resolution * resolution))


# --------------------------------------------------------------------------
# graph construction
# --------------------------------------------------------------------------

class Builder:
    """Keeps node types and depths straight while wiring by hand."""

    def __init__(self, registry, inputs=(), constants=()):
        self.r = registry
        self.nodes = []
        self.types = dict(inputs) | {k: v.type for k, v in constants}
        self.depths = {k: 0 for k, _ in inputs} | {k: -1 for k, _ in constants}
        self.inputs = tuple(inputs)
        self.constants = tuple(constants)

    def add(self, name, op, sources, out=None, params=None, alternatives=()):
        ops = [self.r.resolve(o, tuple(self.types[s] for s in sources), out, params)
               for o in (op, *alternatives)]
        depth = max((self.depths[s] for s in sources), default=0) + 1
        node = Node(name, ops[0].output, tuple(Candidate(o, tuple(sources)) for o in ops),
                    "core", depth, 0 if len(ops) == 1 else None)
        self.nodes.append(node)
        self.types[name] = node.output
        self.depths[name] = depth
        return name

    def choice(self, name, candidates, region="core"):
        """A free node: `candidates` is a list of (operator name, sources, params)."""
        cands = []
        for op, sources, params in candidates:
            cands.append(Candidate(self.r.resolve(op, tuple(self.types[s] for s in sources),
                                                  None, params), tuple(sources)))
        outs = {c.operator.output for c in cands}
        if len(outs) != 1:
            raise TypeError(f"{name}: candidates disagree on output type")
        depth = max((self.depths[s] for c in cands for s in c.sources), default=0) + 1
        node = Node(name, cands[0].operator.output, tuple(cands), region, depth,
                    0 if len(cands) == 1 else None)
        self.nodes.append(node)
        self.types[name] = node.output
        self.depths[name] = depth
        return name

    def program(self, outputs, **kw):
        return Program(self.inputs, tuple(self.nodes), tuple(outputs), self.constants,
                       input_depths=tuple((k, 0) for k, _ in self.inputs), **kw).validate(self.r)


def positional_caller(registry, observation, positions, module_names, index=IDX,
                      port="observation"):
    """Three caller nodes that apply one module at every position of a wide tuple.

    Copied from the unmerged `positional-reuse` branch (`tcn.scaffold.
    positional_scaffold`) so this track does not depend on that branch:

        held    = insert(empty_set_constant, observation)   set[Observation], cap 1
        records = pair(position_constant_set, held)         set[(Index, Observation)]
        mapped  = map(records; module = m)                  set[(Index, Out)]

    `pair` is the cartesian product, so `records` holds one record per position
    carrying the whole observation; the module reads its own position out of the
    record and indexes there.  Three nodes for any number of positions.
    """
    positions = tuple(positions)
    holder = setof(observation, 1)
    locations = setof(index, len(set(positions)))
    hold = registry.resolve("insert", (holder, observation))
    records = registry.resolve("pair", (locations, hold.output))
    ops = [registry.resolve("map", (records.output,), None, {"module": m}) for m in module_names]
    if len({o.output for o in ops}) != 1:
        raise TypeError("modules mapped at one node must share an output interface")
    nodes = (Node("held", hold.output, (Candidate(hold, ("empty", port)),), "core", 1, 0),
             Node("records", records.output, (Candidate(records, ("positions", "held")),), "core", 2, 0),
             Node("mapped", ops[0].output, tuple(Candidate(o, ("records",)) for o in ops),
                  "core", 3, 0 if len(ops) == 1 else None))
    constants = (("positions", Value.of(locations, positions)),
                 ("empty", Value.of(holder, ())))
    return Program(((port, observation),), nodes, (("mapped", "mapped"),), constants,
                   input_depths=((port, 0),)).validate(registry)


# --------------------------------------------------------------------------
# discrete search
# --------------------------------------------------------------------------

def enumerate_reference(program, examples, signals, registry, tolerance=1e-6,
                        max_programs=1 << 24):
    """`tcn.search.enumerate_fit`, with the cost reported as the track requires."""
    n = space_size(program)
    t0 = time.perf_counter()
    res = enumerate_fit(program, examples, signals, registry=registry,
                        tolerance=tolerance, max_programs=max_programs)
    d = res.to_dict()
    d["space_size"] = n
    d["wall"] = time.perf_counter() - t0
    return d


def all_conforming(program, examples, signals, registry, tolerance=1e-6, limit=None):
    """Every conforming discrete program, not just the first.

    `enumerate_fit` reports `unique`; when a solution is *not* unique the useful
    quantity is how many survive, because that is a statement about the probe.
    This is `enumerate_fit`'s own loop reusing its `evaluate`.
    """
    names = [n.name for n in program.nodes]
    counts = [range(len(n.candidates)) for n in program.nodes]
    total = space_size(program)
    t0 = time.perf_counter()
    found, evaluated = [], 0
    for combination in itertools.product(*counts):
        if limit is not None and evaluated >= limit:
            break
        evaluated += 1
        selections = dict(zip(names, combination))
        error = evaluate(program, selections, examples, signals, registry, tolerance)
        if error is not None and error <= tolerance:
            found.append(selections)
    return {"space_size": total, "evaluated": evaluated, "exhausted": evaluated >= total,
            "conforming": found, "count": len(found), "unique": len(found) == 1,
            "seconds": time.perf_counter() - t0}


def random_reference(program, examples, signals, registry, draws, seed=0, tolerance=1e-6):
    import random
    rng = random.Random(seed)
    names = [n.name for n in program.nodes]
    counts = [len(n.candidates) for n in program.nodes]
    t0 = time.perf_counter()
    hits = 0
    for _ in range(draws):
        sel = {k: rng.randrange(c) for k, c in zip(names, counts)}
        err = evaluate(program, sel, examples, signals, registry, tolerance)
        if err is not None and err <= tolerance:
            hits += 1
    return {"draws": draws, "hits": hits, "density": hits / draws,
            "seconds": time.perf_counter() - t0}


def exact_error(program, examples, signals, registry, selections=None):
    worst = 0.
    for ex in examples:
        try:
            _, _, tr = program.execute(ex["inputs"], registry=registry, selections=selections)
        except Exception:
            return float("inf")
        for s in signals:
            a = tr[s.source].flat(); b = ex["targets"][s.target].flat()
            worst = max(worst, max((abs(x - y) for x, y in zip(a, b)), default=0.))
    return worst


def accuracy(program, examples, signals, registry, selections=None, tolerance=1e-6):
    hit = total = 0
    for ex in examples:
        try:
            _, _, tr = program.execute(ex["inputs"], registry=registry, selections=selections)
        except Exception:
            return 0.
        for s in signals:
            a = tr[s.source].flat(); b = ex["targets"][s.target].flat()
            for x, y in zip(a, b):
                total += 1
                hit += abs(x - y) <= tolerance
    return hit / max(1, total)


# --------------------------------------------------------------------------
# gradient reference
# --------------------------------------------------------------------------

def local_fit(program, examples, signals, steps=400, lr=.05, freeze=False, registry=None,
              tolerance=1e-6, polish=0, init_noise=0., seed=0):
    """`tcn.synthesis.fit` with explicit choice-logit initialisation noise.

    `SoftProgram.__init__` uses `torch.zeros`, so two instances over the same
    program are bit-identical and a seed changes nothing.  `init_noise=0`
    reproduces `fit` exactly (see `check_local_fit.py`).
    """
    if not examples:
        raise ValueError("training examples required")
    program.validate_signals(signals)
    model = SoftProgram(program, registry)
    if init_noise:
        g = torch.Generator().manual_seed(int(seed) + 12345)
        with torch.no_grad():
            for p in model.choices:
                p.add_(torch.randn(p.shape, generator=g) * init_noise)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    inputs = {k: torch.stack([_tensor(ex["inputs"][k]) for ex in examples]) for k, _ in program.inputs}
    targets = {s.target: torch.stack([_tensor(ex["targets"][s.target]) for ex in examples])
               for s in signals}

    def loss_fn():
        _, _, tr = model(inputs, return_trace=True)
        return model.probe_loss(tr, targets, signals)

    history = []
    torch.set_num_threads(1)
    for step in range(steps):
        optimizer.zero_grad()
        loss = loss_fn() + .001 * (step / max(1, steps)) * model.entropy()
        if loss.requires_grad:
            loss.backward(); optimizer.step()
        if step % 50 == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.detach())})
    if polish and model.constants:
        held = dict(model.selections()); requires = [p.requires_grad for p in model.choices]
        model.trials = dict(held)
        for p in model.choices: p.requires_grad_(False)
        refiner = torch.optim.Adam([p for p in model.constants.values()], lr=lr)
        for _ in range(polish):
            refiner.zero_grad(); refined = loss_fn()
            if refined.requires_grad: refined.backward(); refiner.step()
        model.trials = {}
        for p, flag in zip(model.choices, requires): p.requires_grad_(flag)
    if freeze:
        scheduler = Crystallizer(model, optimizer, tolerance=tolerance, entropy_limit=.9)
        scheduler.run(loss_fn, rounds=24, retrain_steps=10,
                      conformance=lambda e: exact_error(e, examples, signals, model.registry) <= tolerance)
    err = exact_error(model.export(), examples, signals, model.registry)
    return model, {"training": history, "relaxed_loss": float(loss_fn().detach()),
                   "exact_max_error": err, "exact_conformance": err <= tolerance,
                   "selections": model.selections(), "tolerance": tolerance}


def gradient_arm(build, seeds, examples, signals, registry, steps=400, lr=.05,
                 init_noise=.5, label="", tolerance=1e-6, held=None, verbose=True):
    rows = []
    for seed in seeds:
        program = build()
        t0 = time.perf_counter()
        try:
            model, info = local_fit(program, examples, signals, steps=steps, lr=lr,
                                    registry=registry, tolerance=tolerance,
                                    init_noise=init_noise, seed=seed)
            row = {"seed": seed, "ok": bool(info["exact_conformance"]),
                   "train_err": info["exact_max_error"],
                   "relaxed_loss": info["relaxed_loss"],
                   "selections": info["selections"],
                   "seconds": time.perf_counter() - t0}
            if held is not None:
                row["held_err"] = exact_error(model.export(), held, signals, registry)
                row["held_acc"] = accuracy(model.export(), held, signals, registry)
        except Exception as exc:
            row = {"seed": seed, "ok": False, "train_err": float("inf"),
                   "seconds": time.perf_counter() - t0, "error": repr(exc)[:200]}
        rows.append(row)
        if verbose:
            print(f"  [{label}] seed {seed}: ok={row['ok']} train={row['train_err']:.3g} "
                  f"({row['seconds']:.1f}s) {row.get('error','')}", flush=True)
    return rows


def summarise(rows, key="ok"):
    return {"n": len(rows), "successes": sum(bool(r.get(key)) for r in rows),
            "median_seconds": statistics.median([r["seconds"] for r in rows]) if rows else float("nan"),
            "median_train_err": statistics.median([r["train_err"] for r in rows]) if rows else float("nan")}


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(obj, indent=1, default=str))
    print("wrote", p, flush=True)
    return p


def report(name, value):
    print(f"{name:58s} {value}", flush=True)
