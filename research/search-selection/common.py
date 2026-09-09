"""Cases for the backend selector, and the helpers every script here shares.

A case is a scaffold, its data, and the answer six tracks measured to be right.
Nothing here is a new benchmark: every case is either a shipped fixture, a
scaffold recorded in `research/FINDINGS.md`, or the smallest edit to one of
those that isolates a single property of the rule.

MEASUREMENT NOTE, load-bearing everywhere in this directory. `SoftProgram`
zero-initializes every choice logit, so `torch.manual_seed` does **not** vary
synthesis: without explicit initialization noise a "12-seed" gradient result is
one outcome printed twelve times. Every gradient arm below adds noise through
`perturb()` and says how much; where an arm is run once, it is reported as a
single outcome and not as a rate.
"""
from __future__ import annotations
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
LADDER = ROOT / "research" / "perception-ladder"
if str(LADDER) not in sys.path: sys.path.insert(0, str(LADDER))
OUT = Path(__file__).resolve().parent / "out"

import torch

from tcn.graph import Candidate, Node, Program, Signal
from tcn.learning import SoftProgram, tensor
from tcn.operators import Registry
from tcn.scaffold import F
from tcn.search import enumerate_fit, space_size
from tcn.select import hybrid_fit, select_backend
from tcn.synthesis import fit
from tcn.types import BOOL, Value, product


def dump(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
    print("wrote", p)
    return p


def perturb(model, seed, noise=.5):
    """Explicit initialization noise, because seeds alone vary nothing.

    `SoftProgram.__init__` sets every choice logit to zero, so two runs at
    different `torch.manual_seed` values are bit-identical. Anything reported
    per seed in this directory is perturbed here first.
    """
    g = torch.Generator().manual_seed(seed * 7919 + 13)
    with torch.no_grad():
        for p in model.choices:
            if p.requires_grad: p.add_(torch.randn(p.shape, generator=g) * noise)
    return model


def exact_error(program, examples, signals, registry):
    worst = 0.
    for e in examples:
        try:
            _, _, tr = program.execute(e["inputs"], registry=registry)
        except Exception:
            return float("inf")
        for s in signals:
            a = tr[s.source].flat(); b = e["targets"][s.target].flat()
            worst = max(worst, max((abs(x - y) for x, y in zip(a, b)), default=0.))
    return worst


def selected_error(program, selections, examples, signals, registry):
    return exact_error(program.harden(selections), examples, signals, registry)


@dataclass
class Case:
    name: str
    family: str                 # small-exact | large-space | continuous | dead-surrogate | environment
    program: Program
    signals: tuple
    train: list
    validation: list
    registry: Registry
    right: str                  # the backend the tracks measured to be right
    evidence: str
    rollout_cost: int = 0
    tolerance: float = 1e-3
    test: list = None           # a THIRD split, so held-out is never what was scored
    relax_steps: int = 400
    def __post_init__(self):
        if self.test is None: self.test = []
    @property
    def space(self): return space_size(self.program)
    @property
    def held_out(self):
        """The split a success is judged on: never one the backend was scored on."""
        return self.test or self.validation or self.train
    @property
    def held_out_is_independent(self):
        return bool(self.test) or not self.validation


# ---------------------------------------------------------------- shipped fixtures

def case_mixed():
    from examples.mixed import problem
    p, s, e = problem()
    return Case("mixed", "small-exact", p, s, e, [], Registry(), "enumerate",
                "FINDINGS 7/10: 96 programs swept exhaustively in 41 ms (3.0 ms after the "
                "validation memoization) against a gradient median of 1.5-2.8 s, solution unique, "
                "identical program selected.", tolerance=.005)


def _mixed_constant_problem(k=1.7):
    """`examples/mixed.py` with the analytic scale learned rather than declared.

    This is track 8's failing continuous case, reconstructed: one trainable
    constant on the algebra path. `fit` at the shipped budget converges to
    3.9e-2 rather than to tolerance because the constant's gradient is blurred
    by the candidate mixture, which is exactly the split the hybrid backend
    makes explicit.
    """
    import math as _m
    r = Registry()
    def cands(names, types, sources, output=None):
        return tuple(Candidate(r.resolve(n, types, output), sources) for n in names)
    nodes = (
        Node("logic", BOOL, cands([f"truth_{i}" for i in range(16)], (BOOL, BOOL), ("a", "b")), "logic", 1),
        Node("conversion", F, cands(["encode"], (BOOL,), ("logic",), F), "encoding", 2),
        Node("scaled", F, cands(["mul", "add"], (F, F), ("conversion", "k")), "algebra", 3),
        Node("algebra", F, cands(["add", "sub", "mul"], (F, F), ("scaled", "x")), "algebra", 4),
        Node("analytic", F, cands(["sin", "identity"], (F,), ("algebra",)), "readout", 5))
    p = Program((("a", BOOL), ("b", BOOL), ("x", F)), nodes, (("answer", "analytic"),),
                (("k", Value.of(F, 1.)),), input_depths=(("x", 3),), trainable_constants=("k",))
    signals = (Signal("logic", "logic", ("logic",), BOOL, "bce"),
               Signal("analytic", "answer", ("readout",), F))
    ex = []
    for a in (False, True):
        for b in (False, True):
            for x in (-.7, -.2, .3, .8, -.45, .55):
                z = float(a != b) * k + x
                ex.append({"inputs": {"a": Value.of(BOOL, a), "b": Value.of(BOOL, b), "x": Value.of(F, x)},
                           "targets": {"logic": Value.of(BOOL, a != b), "answer": Value.of(F, _m.sin(z))}})
    return p.validate(r), signals, ex, r


def case_mixed_constant():
    p, s, e, r = _mixed_constant_problem()
    # Split by the continuous input, not by position: the rows are ordered
    # (a, b) outer and x inner, so a positional slice would hand training only
    # half the truth table and make the discrete part undetermined by
    # construction rather than by the search.
    train = [x for i, x in enumerate(e) if i % 6 in (0, 1, 2)]
    val = [x for i, x in enumerate(e) if i % 6 == 3]
    held = [x for i, x in enumerate(e) if i % 6 in (4, 5)]
    return Case("mixed_constant", "continuous", p, s, train, val, r, "hybrid",
                "FINDINGS 10: a single trainable constant misses the conformance tolerance by "
                "3.9e-2 under `fit` because its gradient is blurred by the candidate mixture; "
                "holding the structure and fitting the constant alone is what fixes it.",
                tolerance=1e-3, test=held)


# ---------------------------------------------------------------- the joint scaffold

def _joint_rows(episodes=16):
    """Supervised probe rows for `examples/joint.py`, taken from the generator.

    The two choice nodes are `relation` and `goal_relation`, 16 candidates each:
    the 256-way choice FINDINGS section 7 records brute force settling in
    0.081 ms. The probe targets are the generator's own `gate` and `target`
    channels, read from real episodes rather than assumed.
    """
    from tcn.generation import Host
    from examples.joint import trainer
    t = trainer(1)
    program = t.model.program
    cfg = dict(t.config.generator_config) | {"horizon": t.config.horizon}
    rows = {}
    for i in range(episodes):
        for objective in t.config.objectives:
            h = Host.create("logic", seed=t.config.seed, index=500 + i, split="train",
                            configuration=cfg, objective=objective)
            rec = h.records[-1]
            obs = rec.actor_view().observations
            pr = rec.probes
            key = (tuple(obs["bits"].decoded), obs["goal"].decoded, pr["gate"].decoded, pr["target"].decoded)
            rows[key] = {"inputs": {"bits": obs["bits"], "goal": obs["goal"],
                                    "action": Value.of(product(F, F), (0., 0.)), "dt": Value.of(F, .05)},
                         "targets": {"gate": Value.of(F, float(pr["gate"].decoded)),
                                     "target": Value.of(F, float(pr["target"].decoded))}}
    signals = (Signal("world", "gate", ("encoding",), F), Signal("z", "target", ("encoding",), F))
    return program, signals, list(rows.values()), t


def case_joint_offline(rollout_cost=0):
    program, signals, rows, t = _joint_rows()
    name = "joint_offline" if not rollout_cost else "joint_environment"
    right = "enumerate" if not rollout_cost else "relax"
    evidence = ("FINDINGS 7: the flagship joint result is a 256-way choice brute force settles in "
                "0.081 ms against a 10-36 s gradient run, and the solution is unique."
                if not rollout_cost else
                "FINDINGS 7: at the gradient run's own budget of 704 environment steps the gradient "
                "path succeeds 5/5 where random search over the same 256 candidates succeeds 0/20; "
                "a full enumerative sweep needs 16,384 environment steps.")
    return Case(name, "small-exact" if not rollout_cost else "environment",
                program, signals, rows, [], t.model.registry, right, evidence,
                rollout_cost=rollout_cost, tolerance=1e-3)


# ---------------------------------------------------------------- rung 3, the dead surrogate

def case_rung3(R=2, pool=None, seeds=12, name=None):
    import rung3_geometry as G
    r = Registry()
    pool = (G.BG[0], G.BG[1]) if pool is None else pool
    p = G.centre_program(r, R, free_address=True, pool=pool)
    train = G.examples(range(0, seeds), R)
    val = G.examples(range(100, 100 + seeds), R)
    test = G.examples(range(300, 300 + seeds), R)
    big = len(pool) > 8
    return Case(name or f"rung3_R{R}" + ("_bytes" if big else ""),
                "large-space" if big else "dead-surrogate",
                p, G.centre_signal(), train, val, r,
                "relax" if big else "enumerate",
                ("FINDINGS 14: the full 256-value byte alphabet is 6/6 exact on held-out where brute "
                 "force projects to 107 days."
                 if big else
                 "FINDINGS 11/16: the shipped relaxation is 0/12 exact at 576 and 9,216 programs "
                 "because `eq`'s surrogate is exactly 0.0 at byte spread, while enumeration solves "
                 "every one of them and certifies uniqueness."),
                test=test, relax_steps=800)


# ---------------------------------------------------------------- a gradient boundary

def case_boundary(width=4, name=None):
    """A choice whose only route to the loss runs through `gradient="none"`.

    `pack` is the only operator that takes bytes to a numeric type and it
    declares `gradient="none"`, so the address choice feeding it is not merely
    hard to relax -- it has no path to the objective at all, and `exact_tensor`
    detaches it. FINDINGS section 14 records this as the one place gradient
    descent loses outright. The `scale` node downstream of the boundary is a
    genuine, differentiable choice, so the case also checks that the selector
    distinguishes "this node is cut off" from "this program is cut off".
    """
    from tcn.generation import BYTE
    from tcn.types import integer
    r = Registry()
    BT = product(*(BYTE for _ in range(width)))
    PAIR = product(BYTE, BYTE)
    U16 = integer(16, signed=False)
    nodes = (
        Node("byte", BYTE, tuple(Candidate(r.resolve("project", (BT,), BYTE, {"index": i}), ("raw",))
                                 for i in range(width)), "address", 1),
        Node("tail", BYTE, tuple(Candidate(r.resolve("project", (BT,), BYTE, {"index": i}), ("raw",))
                                 for i in (range(width) if width > 8 else (width - 1,))), "address", 1),
        Node("pair", PAIR, (Candidate(r.resolve("tuple", (BYTE, BYTE)), ("byte", "tail")),), "convert", 2),
        Node("packed", U16, (Candidate(r.resolve("pack", (PAIR,), U16), ("pair",)),), "convert", 3),
        Node("scale", U16, tuple(Candidate(r.resolve(n, (U16, U16)), ("packed", "one")) for n in ("add", "sub", "mul")),
             "algebra", 4))
    p = Program((("raw", BT),), nodes, (("out", "scale"),), (("one", Value.of(U16, 1)),)).validate(r)
    signals = (Signal("scale", "out", ("algebra",), U16),)
    # The target is produced by EXECUTING a reference selection through the same
    # registry, so the reference program is in the space by construction rather
    # than by an assumption about `pack`'s byte order or `add`'s overflow rule.
    reference = {"byte": 2, "tail": 1 if width > 8 else 0, "pair": 0, "packed": 0, "scale": 0}
    rows = []
    for v in range(80):
        raw = tuple((v * (7 * i + 3) + 11 * i + 5) % 254 for i in range(width))
        inputs = {"raw": Value.of(BT, raw)}
        try:
            _, _, trace = p.execute(inputs, registry=r, selections=reference)
        except (OverflowError, ValueError):
            continue                            # an overflowing row is not a target
        rows.append({"inputs": inputs, "targets": {"out": trace["scale"]}})
        if len(rows) == 72: break
    return Case(name or "gradient_boundary", "dead-surrogate", p, signals, rows[:24], rows[24:48], r, "enumerate",
                "FINDINGS 14: relaxation loses exactly where the choice sits behind a "
                "`gradient=\"none\"` boundary; `pack` is that boundary on byte data, and "
                "FINDINGS 11 records enumeration exhausting a 49,152-program `eq` sub-algebra "
                "behind it while gradient descent was 0/8.", test=rows[48:])


# ---------------------------------------------------------------- noisy partial credit

def case_noisy(flips=2):
    """A corrupted 4-input truth table: enumeration recovers, gradient does not.

    FINDINGS section 7 measured minimum-Hamming enumeration recovering the
    uncorrupted function 4/4, 3/4 and 2/4 at 1, 2 and 3 flipped rows against the
    gradient path's 0.50, 0.12 and 0.12, roughly 1000x faster. This is that
    shape at a size the selector can price: a three-gate circuit over four bits
    with `flips` supervision rows inverted.
    """
    r = Registry()
    srcs = (("i0", "i1"), ("i2", "i3"))
    nodes = [Node("g0", BOOL, tuple(Candidate(r.resolve(f"truth_{i}", (BOOL, BOOL)), srcs[0]) for i in range(16)), "logic", 1),
             Node("g1", BOOL, tuple(Candidate(r.resolve(f"truth_{i}", (BOOL, BOOL)), srcs[1]) for i in range(16)), "logic", 1),
             Node("g2", BOOL, tuple(Candidate(r.resolve(f"truth_{i}", (BOOL, BOOL)), ("g0", "g1")) for i in range(16)), "logic", 2)]
    p = Program(tuple((f"i{i}", BOOL) for i in range(4)), tuple(nodes), (("y", "g2"),)).validate(r)
    signals = (Signal("g2", "y", ("logic",), BOOL, "bce"),)
    def reference(a, b, c, d): return (a != b) and (c or d)          # xor / or / and
    rows = []
    for i, bits in enumerate([(a, b, c, d) for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1)]):
        y = reference(*map(bool, bits))
        if i < flips: y = not y                                       # corrupted supervision
        rows.append({"inputs": {f"i{j}": Value.of(BOOL, bool(v)) for j, v in enumerate(bits)},
                     "targets": {"y": Value.of(BOOL, y)}})
    clean = []
    for bits in [(a, b, c, d) for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1)]:
        clean.append({"inputs": {f"i{j}": Value.of(BOOL, bool(v)) for j, v in enumerate(bits)},
                      "targets": {"y": Value.of(BOOL, reference(*map(bool, bits)))}})
    c = Case(f"noisy_{flips}flip", "small-exact", p, signals, rows, [], r, "enumerate",
             "FINDINGS 7: min-Hamming enumeration recovers the uncorrupted function 4/4, 3/4 and "
             "2/4 at 1, 2 and 3 flipped rows against the gradient path's 0.50, 0.12 and 0.12, "
             "roughly 1000x faster. `tcn.search` has no minimum-Hamming objective, so that "
             "advantage is not reachable through `enumerate_fit`; see RESULTS.md.",
             test=clean)
    return c


# ---------------------------------------------------------------- a space too large to sweep

def case_wide_logic(width=5, depth=4):
    """A free-wiring Boolean scaffold whose space is past any sweep budget.

    Nothing exotic: the candidate pool is every `truth_i` over every legal pair
    of predecessors, which is the free-wiring scaffold track 3 measured. At
    width 5 and depth 4 it is 16 * C(k,2) per node and the product runs past
    1e9, so the sweep is priced out and only the relaxation is left.
    """
    r = Registry()
    inputs = tuple((f"i{i}", BOOL) for i in range(width))
    names = [f"i{i}" for i in range(width)]
    nodes = []
    for d in range(depth):
        pool = list(names)
        cands = tuple(Candidate(r.resolve(f"truth_{t}", (BOOL, BOOL)), (a, b))
                      for a in pool for b in pool if a != b for t in (1, 6, 7, 8, 11, 13, 14))
        nodes.append(Node(f"g{d}", BOOL, cands, "logic", d + 1))
        names.append(f"g{d}")
    p = Program(inputs, tuple(nodes), (("y", f"g{depth-1}"),)).validate(r)
    signals = (Signal(f"g{depth-1}", "y", ("logic",), BOOL, "bce"),)
    def reference(v): return ((v[0] != v[1]) and (v[2] or v[3])) != v[4]
    rows = []
    for k in range(1 << width):
        bits = [bool((k >> j) & 1) for j in range(width)]
        rows.append({"inputs": {f"i{j}": Value.of(BOOL, b) for j, b in enumerate(bits)},
                     "targets": {"y": Value.of(BOOL, reference(bits))}})
    return Case("wide_logic", "large-space", p, signals, rows, [], r, "relax",
                "FINDINGS 7/14: past the sweep budget the ordering flips -- enumeration first "
                "fails at depth 5 while the relaxation is the only backend still able to start; "
                "on the 4.3e9-program byte alphabet gradient is 6/6 where brute force projects to "
                "107 days. The 5-input truth table is exhaustive, so training and held-out are the "
                "same 32 rows and this case measures search, not generalization.",
                relax_steps=1200)


CASES = {
    "mixed": case_mixed,
    "mixed_constant": case_mixed_constant,
    "joint_offline": lambda: case_joint_offline(0),
    "joint_environment": lambda: case_joint_offline(rollout_cost=64),
    "rung3_R2": lambda: case_rung3(2),
    "rung3_R4": lambda: case_rung3(4),
    "rung3_bytes": lambda: case_rung3(2, pool=tuple(range(256))),
    "gradient_boundary": case_boundary,
    "boundary_wide": lambda: case_boundary(width=400, name="boundary_wide"),
    "noisy_2flip": lambda: case_noisy(2),
    "wide_logic": case_wide_logic,
}
