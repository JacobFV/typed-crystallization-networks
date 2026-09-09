"""Shared machinery for the byte-numeric track.

The change under test, in one sentence: **`role="byte"` is an *uncommitted*
carrier, and one new representation operator, `interpret`, is the explicit graph
operation that commits it to a magnitude or to a nominal ID.**

Nothing here is domain specific.  The geometry generator supplies pixels because
it is the cheapest source of real bytes in the tree; every type and every
operator used below applies unchanged to a text octet, a file octet or an audio
sample.

Two measurement hazards, both from `research/FINDINGS.md`, are handled here and
named wherever a number depends on them:

* `SoftProgram` zero-initialises every choice logit, so `torch.manual_seed`
  changes nothing.  `local_fit` takes an explicit `init_noise` and every
  gradient number in this track reports it.
* `SoftProgram.__init__` treats **any** node carrying `selected` as frozen and
  runs it through `exact_tensor(...).detach()`.  A one-candidate node built with
  `selected=0` therefore severs the graph.  `Builder` below leaves `selected`
  None by default and only pins where detaching is intended and stated.
"""
from __future__ import annotations

import json
import pathlib
import time

import torch

from tcn.generation import Action, Host
from tcn.graph import Candidate, Node, Program, Signal
from tcn.learning import SoftProgram, tensor as _tensor
from tcn.search import enumerate_fit, evaluate, space_size
from tcn.types import Value, floating, integer, product

# --- types -----------------------------------------------------------------
SCALAR = floating()
VEC3 = product(SCALAR, SCALAR, SCALAR)
BYTE = integer(8, signed=False, role="byte")            # uncommitted octet
MAG8 = integer(8, signed=False, role="intensity")       # committed magnitude
NOM8 = integer(8, signed=False, role="category")        # committed nominal ID
PLAIN8 = integer(8, signed=False)                       # what `pack` produces
FMAG = floating(32, role="intensity")                   # decoded magnitude
FPLAIN = floating(32)                                   # decoded `pack` output
IDX = integer(16, signed=False)

OUT = pathlib.Path(__file__).resolve().parent / "out"


def dump(tag, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}.json").write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def report(label, value):
    print(f"  {label:<62s} {value}", flush=True)


# --- data ------------------------------------------------------------------
def episode(seed, resolution, objects=6, split="train", delta=(-2.4, -2.4, -4.2)):
    """Raw observation only; probes are not read anywhere in this track."""
    h = Host.create("geometry", seed=seed, split=split,
                    configuration={"resolution": resolution, "objects": objects, "horizon": 4})
    rec = h.step([Action("camera", arguments=(("delta", Value.of(VEC3, delta)),))])
    view = rec.actor_view()
    assert set(view.observations) == {"pixels"}, view.observations
    return view.observations["pixels"].decoded[3]


def bytes_type(resolution):
    return product(*(BYTE for _ in range(3 * resolution * resolution)))


def record_type(resolution):
    return product(IDX, bytes_type(resolution))


def block_positions(resolution):
    """Top-left byte address of every 3x3 block fully inside the image."""
    return tuple(3 * (y * resolution + x)
                 for y in range(resolution - 2) for x in range(resolution - 2))


def tap_offset(resolution, dy, dx, channel=1):
    """Byte offset of one stencil tap, relative to the block's top-left pixel."""
    return 3 * (dy * resolution + dx) + channel


STENCIL = tuple((dy, dx) for dy in range(3) for dx in range(3))


def convolve(pixels, resolution, kernel, channel=1):
    """Reference Euclidean convolution, computed outside the substrate.

    `kernel` maps a (dy, dx) stencil tap to a weight.  This is the target the
    learned program must reproduce; it is a plain arithmetic reference, not a
    generator label and not an agent input.
    """
    out = {}
    for start in block_positions(resolution):
        out[start] = float(sum(w * pixels[start + tap_offset(resolution, dy, dx, channel)]
                               for (dy, dx), w in kernel.items()))
    return out


SOBEL_X = {(0, 0): -1., (0, 1): 0., (0, 2): 1.,
           (1, 0): -2., (1, 1): 0., (1, 2): 2.,
           (2, 0): -1., (2, 1): 0., (2, 2): 1.}
DERIVATIVE_X = {(1, 0): -1., (1, 2): 1.}


def examples(seeds, resolution, kernel, split="train", objects=6, out_type=FMAG,
             target="conv", positions=None):
    REC = record_type(resolution)
    rows = []
    for s in seeds:
        pixels = episode(s, resolution, objects=objects, split=split)
        raw = Value.of(bytes_type(resolution), pixels).raw
        labels = convolve(pixels, resolution, kernel)
        for start in (block_positions(resolution) if positions is None else positions):
            rows.append({"inputs": {"rec": Value(REC, (start, raw))},
                         "targets": {target: Value.of(out_type, labels[start])}})
    return rows


def signals(out_type=FMAG, name="conv"):
    return (Signal(name, name, ("core",), out_type, "mse"),)


# --- graph construction ----------------------------------------------------
class Builder:
    """Keeps node types and depths straight while wiring by hand.

    `pin=True` sets `selected`, which `SoftProgram` reads as frozen and executes
    through `exact_tensor(...).detach()`.  That is correct for a node whose value
    is a forward-only argument and wrong for anything a gradient must pass
    through, so it is never the default.
    """

    def __init__(self, registry, inputs=(), constants=(), trainable=()):
        self.r = registry
        self.nodes = []
        self.types = dict(inputs) | {k: v.type for k, v in constants}
        self.depths = {k: 0 for k, _ in inputs} | {k: -1 for k, _ in constants}
        self.inputs = tuple(inputs)
        self.constants = tuple(constants)
        self.trainable = tuple(trainable)

    def add(self, name, op, sources, out=None, params=None, pin=False):
        o = self.r.resolve(op, tuple(self.types[s] for s in sources), out, params)
        depth = max((self.depths[s] for s in sources), default=0) + 1
        self.nodes.append(Node(name, o.output, (Candidate(o, tuple(sources)),), "core", depth,
                               0 if pin else None))
        self.types[name] = o.output
        self.depths[name] = depth
        return name

    def choice(self, name, candidates, region="core"):
        """`candidates` is a list of (operator name, sources, output, params)."""
        cands = []
        for op, sources, out, params in candidates:
            cands.append(Candidate(self.r.resolve(op, tuple(self.types[s] for s in sources),
                                                  out, params), tuple(sources)))
        outs = {c.operator.output for c in cands}
        if len(outs) != 1:
            raise TypeError(f"{name}: candidates disagree on output type")
        depth = max((self.depths[s] for c in cands for s in c.sources), default=0) + 1
        self.nodes.append(Node(name, cands[0].operator.output, tuple(cands), region, depth,
                               0 if len(cands) == 1 else None))
        self.types[name] = cands[0].operator.output
        self.depths[name] = depth
        return name

    def program(self, outputs, **kw):
        return Program(self.inputs, tuple(self.nodes), tuple(outputs), self.constants,
                       input_depths=tuple((k, 0) for k, _ in self.inputs),
                       trainable_constants=self.trainable, **kw).validate(self.r)


# --- evaluation ------------------------------------------------------------
def exact_error(program, rows, sigs, registry, selections=None):
    worst = 0.
    for ex in rows:
        try:
            _, _, tr = program.execute(ex["inputs"], registry=registry, selections=selections)
        except Exception:
            return float("inf")
        for s in sigs:
            a = tr[s.source].flat(); b = ex["targets"][s.target].flat()
            worst = max(worst, max((abs(x - y) for x, y in zip(a, b)), default=0.))
    return worst


def exact_mse(program, rows, sigs, registry, selections=None):
    total = n = 0.
    for ex in rows:
        try:
            _, _, tr = program.execute(ex["inputs"], registry=registry, selections=selections)
        except Exception:
            return float("inf")
        for s in sigs:
            for x, y in zip(tr[s.source].flat(), ex["targets"][s.target].flat()):
                total += (x - y) ** 2; n += 1
    return total / max(1., n)


def target_stats(rows, sigs):
    vals = [v for ex in rows for s in sigs for v in ex["targets"][s.target].flat()]
    mean = sum(vals) / len(vals)
    return {"n": len(vals), "mean": mean, "min": min(vals), "max": max(vals),
            "mse_about_zero": sum(v * v for v in vals) / len(vals),
            "mse_about_mean": sum((v - mean) ** 2 for v in vals) / len(vals)}


def all_conforming(program, rows, sigs, registry, tolerance=1e-6, limit=None):
    """Every conforming discrete selection, not just the first.

    `enumerate_fit` reports whether a solution is unique; when it is not, the
    useful quantity is which ones survive, because that is a statement about the
    supervision rather than about the search.
    """
    import itertools
    names = [n.name for n in program.nodes]
    free = {n.name for n in program.nodes if len(n.candidates) > 1}
    counts = [range(len(n.candidates)) for n in program.nodes]
    found, evaluated = [], 0
    t0 = time.perf_counter()
    for combination in itertools.product(*counts):
        if limit is not None and evaluated >= limit:
            break
        evaluated += 1
        selections = dict(zip(names, combination))
        err = evaluate(program, selections, rows, sigs, registry, tolerance)
        if err is not None and err <= tolerance:
            found.append({k: v for k, v in selections.items() if k in free})
    return {"space_size": space_size(program), "evaluated": evaluated,
            "exhausted": limit is None or evaluated >= space_size(program),
            "count": len(found), "unique": len(found) == 1, "conforming": found,
            "seconds": time.perf_counter() - t0}


def enumerate_reference(program, rows, sigs, registry, tolerance=1e-6, max_programs=1 << 22,
                        rank="order"):
    n = space_size(program)
    t0 = time.perf_counter()
    res = enumerate_fit(program, rows, sigs, registry=registry, tolerance=tolerance,
                        max_programs=max_programs, rank=rank)
    d = res.to_dict(); d["space_size"] = n; d["wall"] = time.perf_counter() - t0
    return d


# --- gradient reference ----------------------------------------------------
def local_fit(program, rows, sigs, steps=400, lr=.05, registry=None, init_noise=0., seed=0,
              entropy_weight=.001, report_grads=(), temperatures=None, constant_noise=0.):
    """`tcn.synthesis.fit`'s control flow plus explicit initialisation noise.

    `SoftProgram` zero-initialises both the choice logits and (through
    `Value.of`) whatever the scaffold declared for a trainable constant, so a
    seed alone changes nothing.  `init_noise` perturbs the logits and
    `constant_noise` the trainable constants; both are reported with every
    number that depends on them.

    `temperatures` maps an operator name to the temperature used at every node
    whose candidates are all that operator, which is how the `index` gather is
    sharpened without disturbing a choice distribution.

    `report_grads` names nodes whose choice-logit gradient is recorded at the
    first backward pass, which is the measurement that distinguishes a live
    conversion boundary from a `gradient="none"` one.
    """
    program.validate_signals(sigs)
    model = SoftProgram(program, registry)
    names = [n.name for n in program.nodes]
    for op_name, tau in dict(temperatures or {}).items():
        for n in program.nodes:
            if all(c.operator.name == op_name for c in n.candidates):
                model.temperatures[n.name] = tau
    g = torch.Generator().manual_seed(int(seed) + 12345)
    if init_noise:
        with torch.no_grad():
            for p in model.choices:
                p.add_(torch.randn(p.shape, generator=g) * init_noise)
    if constant_noise:
        with torch.no_grad():
            for p in model.constants.values():
                p.add_(torch.randn(p.shape, generator=g) * constant_noise)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    inputs = {k: torch.stack([_tensor(ex["inputs"][k]) for ex in rows]) for k, _ in program.inputs}
    targets = {s.target: torch.stack([_tensor(ex["targets"][s.target]) for ex in rows])
               for s in sigs}

    def loss_fn():
        _, _, tr = model(inputs, return_trace=True)
        return model.probe_loss(tr, targets, sigs)

    grads = {}
    history = []
    torch.set_num_threads(1)
    for step in range(steps):
        optimizer.zero_grad()
        loss = loss_fn() + entropy_weight * (step / max(1, steps)) * model.entropy()
        if loss.requires_grad:
            loss.backward()
            if step == 0:
                for name in report_grads:
                    p = model.choices[names.index(name)]
                    grads[name] = None if p.grad is None else float(p.grad.abs().max())
            optimizer.step()
        if step % 50 == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.detach())})
    exported = model.export()
    return model, exported, {"training": history,
                             "relaxed_loss": float(loss_fn().detach()),
                             "choice_gradients": grads,
                             "selections": model.selections(),
                             "init_noise": init_noise, "constant_noise": constant_noise,
                             "seed": seed}
