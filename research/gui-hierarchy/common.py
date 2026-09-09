"""Shared machinery for the `gui-hierarchy` track.

Nothing under `tcn/` or `generators/gui` is modified by anything in this
directory.  Three things are **copied** from `research/discrete-perception`
rather than imported, and all three are named as copies:

* `Builder` -- the hand-wiring helper that keeps node types and depths straight.
* `positional_caller` -- the three-node `insert`/`pair`/`map` pattern from the
  unmerged `positional-reuse` branch (`tcn.scaffold.positional_scaffold`).
* `local_fit` -- `tcn.synthesis.fit`'s control flow plus an `init_noise`
  argument.  `SoftProgram` zero-initialises every choice logit, so
  `torch.manual_seed` does **not** vary synthesis; without explicit noise an
  "N seed" number is one outcome repeated N times.  Every gradient number in this
  track either states its `init_noise` or is reported as a single outcome.

Rules honoured (AGENTS.md): only `record.actor_view().observations` reaches a
program input.  `probes` and `latent_states` build supervision targets and
nothing else.
"""
from __future__ import annotations

import itertools
import json
import pathlib
import statistics
import time

import numpy as np
import torch

from tcn.generation import Host
from tcn.graph import Candidate, Node, Program, Signal
from tcn.learning import SoftProgram, tensor as _tensor
from tcn.operators import Registry
from tcn.search import enumerate_fit, evaluate, space_size
from tcn.types import BOOL, integer, product, setof, Value
from generators.gui.generator import glyphs_from_set, widgets_from_set
from generators.gui.render import KINDS

BYTE = integer(8, signed=False, role="byte")
IDX = integer(16, signed=False)
OUT = pathlib.Path(__file__).resolve().parent / "out"

BASE = {"resolution": 16, "widgets": 6, "nesting": 2, "palette": 32, "horizon": 2}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def episode(seed, split="train", **configuration):
    """Raw observation plus the privileged probes, kept physically separate."""
    configuration = BASE | configuration
    host = Host.create("gui", seed=seed, split=split, configuration=configuration)
    record = host.records[-1]
    view = record.actor_view()
    assert set(view.observations) == {"pixels"}, view.observations
    height, width, channels, pixels = view.observations["pixels"].decoded
    probes = {"hierarchy": widgets_from_set(record.probes["hierarchy"]),
              "owner": [int(v) for v in record.probes["owner"].decoded]}
    if "glyphs" in record.probes:
        probes["glyphs"] = glyphs_from_set(record.probes["glyphs"])
    return {"pixels": pixels, "width": width, "height": height, "channels": channels,
            "probes": probes, "configuration": configuration}


def bytes_type(width, height, channels=3):
    return product(*(BYTE for _ in range(channels * width * height)))


def record_type(width, height, channels=3):
    return product(IDX, bytes_type(width, height, channels))


def colour_at(ep, i):
    return tuple(ep["pixels"][3 * i:3 * i + 3])


def edge_positions(ep):
    """Raster indices whose right and downward neighbours both exist."""
    w, h = ep["width"], ep["height"]
    return tuple(y * w + x for y in range(h - 1) for x in range(w - 1))


def edge_label(ep, i, step=1):
    owner = ep["probes"]["owner"]
    return owner[i] != owner[i + step]


# --------------------------------------------------------------------------
# graph construction (copied from research/discrete-perception/common.py)
# --------------------------------------------------------------------------

class Builder:
    """Keeps node types and depths straight while wiring by hand."""

    def __init__(self, registry, inputs=(), constants=(), relax_single=False):
        # `relax_single` decides whether a node with exactly one candidate is
        # written as `selected=0`.  `SoftProgram.forward` treats any pre-selected
        # node as frozen and evaluates it through `exact_tensor(...).detach()`,
        # which severs the autograd graph there -- so a single-candidate node on
        # the path between a choice and the loss makes that choice's logit
        # unreachable.  Leaving `selected=None` keeps the node relaxed at no
        # change to the discrete space (a product over a 1-element choice).  This
        # is a scaffold-side option, not a change to `tcn/`.
        self.relax_single = relax_single
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
                    "core", depth, None if self.relax_single or len(ops) > 1 else 0)
        self.nodes.append(node)
        self.types[name] = node.output
        self.depths[name] = depth
        return name

    def choice(self, name, candidates, region="core"):
        cands = []
        for op, sources, params in candidates:
            cands.append(Candidate(self.r.resolve(op, tuple(self.types[s] for s in sources),
                                                  None, params), tuple(sources)))
        outs = {c.operator.output for c in cands}
        if len(outs) != 1:
            raise TypeError(f"{name}: candidates disagree on output type")
        depth = max((self.depths[s] for c in cands for s in c.sources), default=0) + 1
        node = Node(name, cands[0].operator.output, tuple(cands), region, depth,
                    None if self.relax_single or len(cands) > 1 else 0)
        self.nodes.append(node)
        self.types[name] = node.output
        self.depths[name] = depth
        return name

    def program(self, outputs, **kw):
        return Program(self.inputs, tuple(self.nodes), tuple(outputs), self.constants,
                       input_depths=tuple((k, 0) for k, _ in self.inputs), **kw).validate(self.r)


def positional_caller(registry, observation, positions, module_names, index=IDX,
                      port="observation"):
    """Three caller nodes applying one module at every position of a wide tuple.

    Copied from the unmerged `positional-reuse` branch
    (`tcn.scaffold.positional_scaffold`) so this track does not depend on it.
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
             Node("records", records.output, (Candidate(records, ("positions", "held")),),
                  "core", 2, 0),
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
    started = time.perf_counter()
    result = enumerate_fit(program, examples, signals, registry=registry, tolerance=tolerance,
                           max_programs=max_programs).to_dict()
    result["space_size"] = space_size(program)
    result["wall"] = time.perf_counter() - started
    result["node_evaluations"] = result["evaluated"] * len(program.nodes) * len(examples)
    return result


def all_conforming(program, examples, signals, registry, tolerance=1e-6, limit=None):
    """Every conforming discrete program, not just the lexicographically first.

    `enumerate_fit` reports `unique`; when a solution is not unique the useful
    quantity is how many survive, because that is a statement about the probe.
    """
    names = [n.name for n in program.nodes]
    counts = [range(len(n.candidates)) for n in program.nodes]
    total = space_size(program)
    started = time.perf_counter()
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
            "seconds": time.perf_counter() - started,
            "node_evaluations": evaluated * len(program.nodes) * len(examples)}


def random_reference(program, examples, signals, registry, draws, seed=0, tolerance=1e-6):
    import random
    rng = random.Random(seed)
    names = [n.name for n in program.nodes]
    counts = [len(n.candidates) for n in program.nodes]
    started = time.perf_counter()
    hits = 0
    for _ in range(draws):
        selections = {k: rng.randrange(c) for k, c in zip(names, counts)}
        error = evaluate(program, selections, examples, signals, registry, tolerance)
        hits += error is not None and error <= tolerance
    return {"draws": draws, "hits": hits, "density": hits / draws,
            "seconds": time.perf_counter() - started}


def exact_error(program, examples, signals, registry, selections=None):
    worst = 0.
    for example in examples:
        try:
            _, _, trace = program.execute(example["inputs"], registry=registry,
                                          selections=selections)
        except Exception:
            return float("inf")
        for signal in signals:
            a = trace[signal.source].flat()
            b = example["targets"][signal.target].flat()
            worst = max(worst, max((abs(x - y) for x, y in zip(a, b)), default=0.))
    return worst


def accuracy(program, examples, signals, registry, selections=None, tolerance=1e-6):
    hit = total = 0
    for example in examples:
        try:
            _, _, trace = program.execute(example["inputs"], registry=registry,
                                          selections=selections)
        except Exception:
            return 0.
        for signal in signals:
            a = trace[signal.source].flat()
            b = example["targets"][signal.target].flat()
            for x, y in zip(a, b):
                total += 1
                hit += abs(x - y) <= tolerance
    return hit / max(1, total)


# --------------------------------------------------------------------------
# gradient reference (copied from research/discrete-perception/common.py)
# --------------------------------------------------------------------------

def local_fit(program, examples, signals, steps=400, lr=.05, registry=None, tolerance=1e-6,
              polish=0, init_noise=0., seed=0):
    """`tcn.synthesis.fit` with explicit choice-logit initialisation noise.

    `SoftProgram.__init__` uses `torch.zeros`, so two instances over the same
    program are bit-identical and a seed changes nothing.  `init_noise=0`
    reproduces `fit` exactly.
    """
    if not examples:
        raise ValueError("training examples required")
    program.validate_signals(signals)
    model = SoftProgram(program, registry)
    if init_noise:
        generator = torch.Generator().manual_seed(int(seed) + 12345)
        with torch.no_grad():
            for p in model.choices:
                p.add_(torch.randn(p.shape, generator=generator) * init_noise)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    inputs = {k: torch.stack([_tensor(ex["inputs"][k]) for ex in examples])
              for k, _ in program.inputs}
    targets = {s.target: torch.stack([_tensor(ex["targets"][s.target]) for ex in examples])
               for s in signals}

    def loss_fn():
        _, _, trace = model(inputs, return_trace=True)
        return model.probe_loss(trace, targets, signals)

    history = []
    torch.set_num_threads(1)
    for step in range(steps):
        optimizer.zero_grad()
        loss = loss_fn() + .001 * (step / max(1, steps)) * model.entropy()
        if loss.requires_grad:
            loss.backward()
            optimizer.step()
        if step % 50 == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.detach())})
    if polish and model.constants:
        held = dict(model.selections())
        requires = [p.requires_grad for p in model.choices]
        model.trials = dict(held)
        for p in model.choices:
            p.requires_grad_(False)
        refiner = torch.optim.Adam([p for p in model.constants.values()], lr=lr)
        for _ in range(polish):
            refiner.zero_grad()
            refined = loss_fn()
            if refined.requires_grad:
                refined.backward()
                refiner.step()
        model.trials = {}
        for p, flag in zip(model.choices, requires):
            p.requires_grad_(flag)
    error = exact_error(model.export(), examples, signals, model.registry)
    return model, {"training": history, "relaxed_loss": float(loss_fn().detach()),
                   "exact_max_error": error, "exact_conformance": error <= tolerance,
                   "selections": model.selections(), "tolerance": tolerance}


def gradient_arm(build, seeds, examples, signals, registry, steps=400, lr=.15, init_noise=.5,
                 label="", tolerance=1e-6, held=None, verbose=True):
    rows = []
    for seed in seeds:
        program = build()
        started = time.perf_counter()
        try:
            model, info = local_fit(program, examples, signals, steps=steps, lr=lr,
                                    registry=registry, tolerance=tolerance,
                                    init_noise=init_noise, seed=seed)
            row = {"seed": seed, "ok": bool(info["exact_conformance"]),
                   "train_err": info["exact_max_error"], "relaxed_loss": info["relaxed_loss"],
                   "selections": info["selections"], "seconds": time.perf_counter() - started}
            if held is not None:
                row["held_err"] = exact_error(model.export(), held, signals, registry)
                row["held_acc"] = accuracy(model.export(), held, signals, registry)
        except Exception as exc:
            row = {"seed": seed, "ok": False, "train_err": float("inf"),
                   "seconds": time.perf_counter() - started, "error": repr(exc)[:200]}
        rows.append(row)
        if verbose:
            print(f"  [{label}] seed {seed}: ok={row['ok']} train={row['train_err']:.3g} "
                  f"held={row.get('held_err')} ({row['seconds']:.1f}s) {row.get('error','')}",
                  flush=True)
    return rows


def summarise(rows, key="ok"):
    return {"n": len(rows), "successes": sum(bool(r.get(key)) for r in rows),
            "held_exact": sum(r.get("held_err") == 0. for r in rows),
            "median_seconds": statistics.median([r["seconds"] for r in rows]) if rows else None,
            "median_train_err": statistics.median([r["train_err"] for r in rows]) if rows else None}


# --------------------------------------------------------------------------
# recoverability
# --------------------------------------------------------------------------

def lookup_bound(items, evaluation):
    """Best possible predictor given exactly a context, two ways.

    `items` and `evaluation` are lists of `(context, label)`.  The method is the
    object-identity track's: a lookup table keyed by the context is an upper
    bound on *every* function of that context, so if it does no better than the
    majority label, no program over that context can.

    * `oracle` fits the table on the evaluation set itself -- the Bayes accuracy
      of the context on that distribution, and therefore a hard ceiling.
    * `transfer` fits on `items` and scores `evaluation`, falling back to the
      training majority on a key never seen -- what a *learned* table achieves.
    """
    def table(rows):
        counts = {}
        for context, label in rows:
            counts.setdefault(context, {}).setdefault(label, 0)
            counts[context][label] += 1
        return {k: max(v.items(), key=lambda kv: (kv[1], str(kv[0])))[0] for k, v in counts.items()}

    labels = [label for _, label in evaluation]
    majority = max(set(labels), key=labels.count) if labels else None
    baseline = sum(label == majority for label in labels) / max(1, len(labels))
    fitted = table(items)
    train_labels = [label for _, label in items]
    fallback = max(set(train_labels), key=train_labels.count) if train_labels else None
    unseen = sum(context not in fitted for context, _ in evaluation)
    transfer = sum(fitted.get(context, fallback) == label for context, label in evaluation)
    oracle_table = table(evaluation)
    oracle = sum(oracle_table[context] == label for context, label in evaluation)
    return {"train_records": len(items), "eval_records": len(evaluation),
            "distinct_train_keys": len(fitted), "distinct_eval_keys": len(oracle_table),
            "unseen_key_fraction": unseen / max(1, len(evaluation)),
            "majority_baseline": baseline,
            "oracle_accuracy": oracle / max(1, len(evaluation)),
            "oracle_advantage": oracle / max(1, len(evaluation)) - baseline,
            "transfer_accuracy": transfer / max(1, len(evaluation)),
            "transfer_advantage": transfer / max(1, len(evaluation)) - baseline}


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(obj, indent=1, default=str))
    print("wrote", path, flush=True)
    return path


def report(name, value):
    print(f"{name:62s} {value}", flush=True)
