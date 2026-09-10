"""Shared machinery for the `visual-ladder` track (rungs two and three).

This track continues `research/gui-hierarchy` directly.  Nothing under `tcn/` is
modified; `generators/gui/` is extended only behind a default-off configuration
key (`label_alphabet`), and the default stream is verified byte-identical.

Three things are copied rather than imported, and all three are named as copies:

* `Builder` -- the hand-wiring helper from `research/gui-hierarchy/common.py`,
  extended with `trainable` (from `research/byte-numeric/common.py`) so a node
  pool and a continuous weight can live in the same scaffold.
* `local_fit` -- `tcn.synthesis.fit`'s control flow plus `init_noise`,
  `constant_noise` and a per-operator `temperatures` override.  `SoftProgram`
  zero-initialises every choice logit *and* every trainable constant, so
  `torch.manual_seed` alone does not vary synthesis and an "N seed" number
  without explicit noise is one outcome repeated N times.
* `lookup_bound` -- the object-identity track's recoverability ceiling.

`tcn.scaffold.positional_scaffold` and `tcn.search.enumerate_prefix` are on main
and are imported, not copied.

Rules honoured (AGENTS.md): only `record.actor_view().observations` ever reaches
a program input.  `probes` and `latent_states` build supervision targets and
nothing else, and every `episode()` asserts the observation set.
"""
from __future__ import annotations

import json
import pathlib
import statistics
import time

import torch

from tcn.generation import Host
from tcn.graph import Candidate, Node, Program, Signal
from tcn.learning import SoftProgram, tensor as _tensor
from tcn.operators import Registry
from tcn.scaffold import positional_scaffold
from tcn.search import enumerate_prefix, evaluate, space_size
from tcn.types import BOOL, Value, floating, integer, product, setof
from generators.gui.generator import glyphs_from_set, widgets_from_set
from generators.gui.render import ALPHABET, INK, KINDS

# --- types -----------------------------------------------------------------
BYTE = integer(8, signed=False, role="byte")        # uncommitted octet, as rendered
MAG8 = integer(8, signed=False, role="intensity")   # committed magnitude
PLAIN8 = integer(8, signed=False)                   # plain integer carrier
FMAG = floating(32, role="intensity")               # decoded magnitude
IDX = integer(16, signed=False)                     # a computed address
FIELD = integer(8, signed=False)                    # the probe's coordinate field

OUT = pathlib.Path(__file__).resolve().parent / "out"

# The screens.  Every number in this track is stated against one of these, since
# `research/gui-hierarchy` measured that the request is not the achievement.
FLAT = {"resolution": 32, "widgets": 20, "nesting": 5, "min_size": 4, "palette": 32,
        "horizon": 2}
TEXT = {"resolution": 64, "widgets": 24, "nesting": 6, "min_size": 6, "palette": 48,
        "horizon": 2, "labels": True, "label_size": 8, "hierarchy_capacity": 32,
        "glyph_capacity": 64}


# --- data ------------------------------------------------------------------
def episode(seed, split="train", **configuration):
    """Raw observation plus the privileged probes, kept physically separate."""
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
            "probes": probes, "configuration": dict(configuration), "seed": seed,
            "split": split}


def bytes_type(width, height, channels=3):
    return product(*(BYTE for _ in range(channels * width * height)))


def address_type(width, height, channels=3):
    """`IDX` refined to the raster it addresses: exactly the legal byte offsets.

    `research/refinement-bounds` only.  Nothing in this track builds it by
    default -- every entry point below keeps `IDX` unless an `addr` is passed --
    so declaring it changes no existing artifact, search or certificate.

    The upper endpoint is the **last byte** `3WH - 1`, not the last *pixel*
    `3WH - 3`, because `Registry.resolve` gives a binary operator the same output
    type as its operands, so a bound declared on an address `a` is inherited by
    the `a + 1` and `a + 2` that `same_scaffold` computes from it.  A bound of
    `(0, 3WH - 3)` -- the value the `min(., 3069)` clamp enforces -- is violated
    by `a + 2` on a reachable input.  See `research/refinement-bounds/RESULTS.md`.
    """
    return integer(16, signed=False, bounds=(0, channels * width * height - 1))


def record_type(width, height, channels=3, addr=None):
    return product(addr or IDX, bytes_type(width, height, channels))


def colour_at(ep, x, y):
    i = 3 * (y * ep["width"] + x)
    return tuple(ep["pixels"][i:i + 3])


def is_ink(ep, x, y):
    return colour_at(ep, x, y) == INK


def ink_grid(ep):
    """Per-pixel ink indicator -- the reduction the R2 bound identified.

    `eq(pixel_channel, 0)` on all three channels; expressible as three `eq` nodes
    and two `and`s, and it is what takes the glyph bound's transfer from 0.0875
    to 0.9250 in `research/gui-hierarchy`.
    """
    w, h = ep["width"], ep["height"]
    return [[1 if is_ink(ep, x, y) else 0 for x in range(w)] for y in range(h)]


def window_bits(grid, x0, y0, ww, hh, width, height):
    """`ww * hh` ink bits with the window's top-left at (x0, y0); off-screen is 0."""
    return tuple(grid[y0 + dy][x0 + dx] if 0 <= y0 + dy < height and 0 <= x0 + dx < width else 0
                 for dy in range(hh) for dx in range(ww))


def window_bytes(ep, x0, y0, ww, hh):
    """The raw bytes of the same window, for the raw-vs-ink transfer comparison."""
    out = []
    for dy in range(hh):
        for dx in range(ww):
            x, y = x0 + dx, y0 + dy
            out.extend(colour_at(ep, x, y) if 0 <= x < ep["width"] and 0 <= y < ep["height"]
                       else (0, 0, 0))
    return tuple(out)


# --- graph construction (copied from research/gui-hierarchy/common.py) ------
class Builder:
    """Keeps node types and depths straight while wiring by hand.

    `relax_single=True` leaves a one-candidate node's `selected` as None.
    `SoftProgram.forward` used to treat any pre-selected node as frozen and
    evaluate it detached, severing every upstream choice; that is fixed on main,
    but the option is kept so the two behaviours stay comparable.
    """

    def __init__(self, registry, inputs=(), constants=(), trainable=(), relax_single=True):
        self.r = registry
        self.nodes = []
        self.types = dict(inputs) | {k: v.type for k, v in constants}
        self.depths = {k: 0 for k, _ in inputs} | {k: -1 for k, _ in constants}
        self.inputs = tuple(inputs)
        self.constants = tuple(constants)
        self.trainable = tuple(trainable)
        self.relax_single = relax_single

    def add(self, name, op, sources, out=None, params=None):
        o = self.r.resolve(op, tuple(self.types[s] for s in sources), out, params)
        depth = max((self.depths[s] for s in sources), default=0) + 1
        self.nodes.append(Node(name, o.output, (Candidate(o, tuple(sources)),), "core", depth,
                               None if self.relax_single else 0))
        self.types[name] = o.output
        self.depths[name] = depth
        return name

    def choice(self, name, candidates, region="core"):
        """`candidates` is a list of (operator, sources, output, params)."""
        cands = []
        for entry in candidates:
            op, sources, out, params = (tuple(entry) + (None, None))[:4]
            cands.append(Candidate(self.r.resolve(op, tuple(self.types[s] for s in sources),
                                                  out, params), tuple(sources)))
        outs = {c.operator.output for c in cands}
        if len(outs) != 1:
            raise TypeError(f"{name}: candidates disagree on output type")
        depth = max((self.depths[s] for c in cands for s in c.sources), default=0) + 1
        self.nodes.append(Node(name, cands[0].operator.output, tuple(cands), region, depth,
                               None if self.relax_single or len(cands) > 1 else 0))
        self.types[name] = cands[0].operator.output
        self.depths[name] = depth
        return name

    def program(self, outputs, **kw):
        return Program(self.inputs, tuple(self.nodes), tuple(outputs), self.constants,
                       input_depths=tuple((k, 0) for k, _ in self.inputs),
                       trainable_constants=self.trainable, **kw).validate(self.r)


def caller(registry, observation, positions, module_names, index=IDX, port="observation"):
    """`tcn.scaffold.positional_scaffold`: three nodes, any number of positions."""
    return positional_scaffold(registry, observation, positions, module_names, index=index,
                               port=port)


# --- discrete search -------------------------------------------------------
def sweep(program, examples, signals, registry, tolerance=1e-6, max_programs=1 << 22):
    """`enumerate_prefix` -- exhaustive, certificate preserved, prefix-reusing.

    The default fast path recommended by `research/discrete-backend`: identical
    conforming set to `enumerate_fit` at 77x the speed, and `SearchResult` now
    carries `conforming` and an explicit `certificate`, so this track never has
    to reimplement the sweep to learn how many programs survived.
    """
    started = time.perf_counter()
    result = enumerate_prefix(program, examples, signals, registry=registry,
                              tolerance=tolerance, max_programs=max_programs).to_dict()
    result["space_size"] = space_size(program)
    result["wall"] = time.perf_counter() - started
    return result


def all_conforming(program, examples, signals, registry, tolerance=1e-6, max_programs=1 << 22):
    """Every conforming selection, walked with prefix reuse.

    `enumerate_prefix` returns only the chosen one, so the walk is repeated here
    with a collector.  It is the same DFS; `conforming` from `sweep` is the
    cross-check that this reproduces it.
    """
    import itertools
    names = [n.name for n in program.nodes]
    counts = [range(len(n.candidates)) for n in program.nodes]
    total = space_size(program)
    started = time.perf_counter()
    found, evaluated = [], 0
    for combination in itertools.product(*counts):
        if evaluated >= max_programs:
            break
        evaluated += 1
        selections = dict(zip(names, combination))
        error = evaluate(program, selections, examples, signals, registry, tolerance)
        if error is not None and error <= tolerance:
            found.append(selections)
    return {"space_size": total, "evaluated": evaluated, "exhausted": evaluated >= total,
            "conforming": found, "count": len(found), "unique": len(found) == 1,
            "seconds": time.perf_counter() - started}


def random_reference(program, examples, signals, registry, draws, seed=0, tolerance=1e-6):
    import random
    rng = random.Random(seed)
    names = [n.name for n in program.nodes]
    counts = [len(n.candidates) for n in program.nodes]
    hits = 0
    for _ in range(draws):
        selections = {k: rng.randrange(c) for k, c in zip(names, counts)}
        error = evaluate(program, selections, examples, signals, registry, tolerance)
        hits += error is not None and error <= tolerance
    return {"draws": draws, "hits": hits, "density": hits / draws}


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


# --- gradient reference ----------------------------------------------------
def local_fit(program, rows, signals, steps=400, lr=.05, registry=None, tolerance=1e-6,
              init_noise=0., constant_noise=0., seed=0, temperatures=None, entropy_weight=.001,
              report_grads=()):
    """`tcn.synthesis.fit`'s control flow plus explicit initialisation noise.

    `temperatures` maps an operator name to the temperature used at every node
    all of whose candidates are that operator.  That is how the `index` gather is
    sharpened without disturbing any choice distribution -- those nodes have one
    candidate, so a 1-way softmax is temperature-invariant.  On merged main the
    candidate and relaxation temperatures are separable (`surrogate_scale`), and
    where this track uses that it says so.
    """
    program.validate_signals(signals)
    model = SoftProgram(program, registry)
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
               for s in signals}

    def loss_fn():
        _, _, tr = model(inputs, return_trace=True)
        return model.probe_loss(tr, targets, signals)

    history, grads = [], {}
    torch.set_num_threads(1)
    for step in range(steps):
        optimizer.zero_grad()
        loss = loss_fn() + entropy_weight * (step / max(1, steps)) * model.entropy()
        if loss.requires_grad:
            loss.backward()
            if step == 0:
                for name in report_grads:
                    p = model.trial(name) if hasattr(model, "trial") else None
                    grads[name] = None
                optimizer.step()
            else:
                optimizer.step()
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.detach())})
    exported = model.export()
    error = exact_error(exported, rows, signals, model.registry)
    return model, {"training": history, "relaxed_loss": float(loss_fn().detach()),
                   "exact_max_error": error, "exact_conformance": error <= tolerance,
                   "selections": model.selections(), "tolerance": tolerance}


def gradient_arm(build, seeds, rows, signals, registry, steps=300, lr=.05, init_noise=0.,
                 constant_noise=1., label="", tolerance=1e-6, held=None, temperatures=None,
                 verbose=True, extra=None):
    out = []
    for seed in seeds:
        program = build()
        started = time.perf_counter()
        try:
            model, info = local_fit(program, rows, signals, steps=steps, lr=lr,
                                    registry=registry, tolerance=tolerance,
                                    init_noise=init_noise, constant_noise=constant_noise,
                                    seed=seed, temperatures=temperatures)
            row = {"seed": seed, "ok": bool(info["exact_conformance"]),
                   "train_err": info["exact_max_error"], "relaxed_loss": info["relaxed_loss"],
                   "selections": info["selections"], "seconds": time.perf_counter() - started}
            exported = model.export()
            if held is not None:
                row["held_err"] = exact_error(exported, held, signals, registry)
                row["held_acc"] = accuracy(exported, held, signals, registry)
            if extra is not None:
                row.update(extra(model, exported))
        except Exception as exc:
            row = {"seed": seed, "ok": False, "train_err": float("inf"),
                   "seconds": time.perf_counter() - started, "error": repr(exc)[:300]}
        out.append(row)
        if verbose:
            print(f"  [{label}] seed {seed}: ok={row['ok']} train={row['train_err']:.4g} "
                  f"held={row.get('held_err')} acc={row.get('held_acc')} "
                  f"({row['seconds']:.1f}s) {row.get('error','')}", flush=True)
    return out


def summarise(rows, key="ok"):
    good = [r for r in rows if r.get("train_err") not in (None, float("inf"))]
    return {"n": len(rows), "successes": sum(bool(r.get(key)) for r in rows),
            "held_exact": sum(r.get("held_err") == 0. for r in rows),
            "median_held_acc": statistics.median([r["held_acc"] for r in rows
                                                  if "held_acc" in r]) if any(
                                                      "held_acc" in r for r in rows) else None,
            "median_seconds": statistics.median([r["seconds"] for r in rows]) if rows else None,
            "median_train_err": statistics.median([r["train_err"] for r in good]) if good else None}


# --- recoverability --------------------------------------------------------
def lookup_bound(items, evaluation):
    """Best possible predictor given exactly a context, two ways.

    Copied from `research/gui-hierarchy/common.py`, itself the object-identity
    track's method: a lookup table keyed by the context is an upper bound on
    *every* function of that context, so if it does no better than the majority
    label, no program over that context can.

    * `oracle` fits on the evaluation set itself -- a hard ceiling.
    * `transfer` fits on `items` and scores `evaluation`, falling back to the
      training majority on an unseen key -- what a learned table achieves.
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
    path.write_text(json.dumps(obj, indent=1, default=str, sort_keys=True))
    print("wrote", path, flush=True)
    return path


def load(name):
    return json.loads((OUT / f"{name}.json").read_text())


def report(name, value):
    print(f"{name:66s} {value}", flush=True)
