"""A minimal, controlled instrument for the address-relaxation wall.

One node, one address choice, a known reference address, and a landscape small
enough to map exhaustively.  Nothing here touches `tcn/` or `generators/`; the
data are generated in this file so that the two properties the stated
explanation is about -- how correlated neighbouring addresses are, and how
different their marginal means are -- can be dialled independently.

Two mechanisms are instrumented, because TCN has two ways to spell an address:

  DISCRETE  one node whose candidates are `project(arr, index=i)` for every i.
            `SoftProgram` mixes them with `softmax(logits)`.  This is exactly
            what `graph.legal_candidates` produces for a tuple-typed port, and
            it is the mechanism the perception ladder measured.

  INDEX     one node `index(arr, addr)` with `addr` a trainable constant.
            `learning.relaxed` softmaxes `-(addr - arange)^2 / temperature`
            over positions and mixes the addressed values.  This is TCN's
            existing continuous address relaxation.

Both are measured under two downstreams:

  DIRECT    the target is the addressed value itself.  The relaxed loss is then
            an exact convex quadratic in the mixture weights, which makes the
            landscape analysable in closed form.
  EQ        the target is `eq(addressed_value, constant)`.  The address sits
            behind a nonlinearity, which is the perception ladder's actual
            shape.

MEASUREMENT NOTES, honoured throughout:
  * `SoftProgram` zero-initialises every choice logit, so `torch.manual_seed`
    does not vary synthesis.  Every statistic below varies the *data* seed and,
    where logits are perturbed, adds explicit noise with a stated scale.  No
    number here is a seed statistic over identical initialisations.
  * `relaxed`'s `tuple` is a bare `torch.cat` that will not broadcast a batched
    value against an unbatched constant, so nothing here tuples a constant with
    an intermediate.
  * Node evaluations are counted alongside wall clock; the host is shared.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from tcn.graph import Candidate, Node, Program, Signal
from tcn.learning import SoftProgram, exact_tensor, relaxed, tensor
from tcn.operators import Registry
from tcn.types import BOOL, Type, Value, floating, integer, product

OUT = Path(__file__).resolve().parent / "out"

V = floating(32)                      # the addressed element type
ADDR = integer(16, signed=False)      # the address carrier for `index`

torch.set_num_threads(1)


# --------------------------------------------------------------------------
# evaluation budget
# --------------------------------------------------------------------------

@dataclass
class Budget:
    """Node evaluations and wall clock, reported together as required."""
    node_evals: int = 0
    seconds: float = 0.0
    _t0: float = field(default_factory=time.perf_counter)

    def add(self, n=1):
        self.node_evals += n

    def stop(self):
        self.seconds = time.perf_counter() - self._t0
        return self

    def to_dict(self):
        return {"node_evals": self.node_evals, "seconds": round(self.seconds, 3)}


BUDGET = Budget()


# --------------------------------------------------------------------------
# data regimes
# --------------------------------------------------------------------------

def gp_arrays(n_positions, n_examples, corr_length, mean_spread, seed,
              scale=1.0, offset=0.0):
    """Arrays with independently controllable neighbour correlation and mean spread.

    `corr_length` is the length scale of a Gaussian kernel over positions: 0.0
    gives independent addresses (the "unrelated values" regime the stated
    explanation is about), large values give a smoothly varying array where
    neighbouring addresses hold similar values.

    `mean_spread` is the standard deviation of a per-address constant offset,
    fixed across examples.  It leaves the *correlation* structure of the
    fluctuations untouched and only makes the addresses differ in mean.
    """
    g = torch.Generator().manual_seed(int(seed))
    idx = torch.arange(n_positions, dtype=torch.float64)
    if corr_length <= 0:
        cov = torch.eye(n_positions, dtype=torch.float64)
    else:
        d = (idx[:, None] - idx[None, :]) ** 2
        cov = torch.exp(-d / (2 * corr_length ** 2))
    cov = cov + 1e-6 * torch.eye(n_positions, dtype=torch.float64)
    chol = torch.linalg.cholesky(cov)
    z = torch.randn(n_examples, n_positions, dtype=torch.float64, generator=g)
    x = z @ chol.T
    mu = torch.randn(n_positions, dtype=torch.float64, generator=g) * mean_spread
    return (x + mu) * scale + offset


REGIMES = {
    # name: (corr_length, mean_spread, scale, offset)
    "iid_centred":        (0.0, 0.0, 1.0, 0.0),
    "smooth_centred":     (3.0, 0.0, 1.0, 0.0),
    "iid_mean_spread":    (0.0, 3.0, 1.0, 0.0),
    "smooth_mean_spread": (3.0, 3.0, 1.0, 0.0),
    # a byte-like regime: positive, large mean, large per-address mean spread
    "bytelike":           (0.0, 40.0, 1.0, 128.0),
    "bytelike_smooth":    (3.0, 40.0, 1.0, 128.0),
}

# `eq`'s surrogate is `exp(-(a-b)^2/tau)` at tau=1, which is exactly 0.0 in
# float32 for |a-b| >= 11 (FINDINGS section 11).  A byte-scale alphabet
# therefore kills the gradient for reasons that have nothing to do with
# addressing, so the `eq` arms use a small integer alphabet where the surrogate
# is alive at every pair of values, and the underflow is reported separately.
EQ_REGIMES = {
    # name: (corr_length, mean_spread_in_levels)
    "iid_centred":        (0.0, 0.0),
    "smooth_centred":     (3.0, 0.0),
    "iid_mean_spread":    (0.0, 1.5),
    "smooth_mean_spread": (3.0, 1.5),
}


def eq_arrays(regime, n_positions, n_examples, seed):
    """Small-alphabet integer arrays for the `eq` downstream."""
    corr, spread = EQ_REGIMES[regime]
    a = gp_arrays(n_positions, n_examples, corr, 0.0, seed=seed)
    a = torch.round(a).clamp(-3, 3)
    if spread:
        g = torch.Generator().manual_seed(int(seed) + 5)
        mu = torch.round(torch.randn(n_positions, generator=g, dtype=torch.float64) * spread)
        a = a + mu
    return a


def make_examples(arrays, k, downstream="direct", eq_constant=None):
    """Wrap raw arrays as TCN examples with a known reference address `k`."""
    n = arrays.shape[1]
    arr_t = product(*([V] * n))
    ex = []
    for row in arrays.tolist():
        inputs = {"arr": Value.of(arr_t, tuple(float(v) for v in row))}
        if downstream == "direct":
            targets = {"y": Value.of(V, float(row[k]))}
        else:
            targets = {"y": Value.of(BOOL, bool(abs(row[k] - eq_constant) < 1e-9))}
        ex.append({"inputs": inputs, "targets": targets})
    return ex


# --------------------------------------------------------------------------
# programs
# --------------------------------------------------------------------------

def discrete_program(registry, n, downstream="direct", eq_constant=0.0):
    """One node whose candidates are the n possible addresses (`project`)."""
    arr_t = product(*([V] * n))
    cands = tuple(Candidate(registry.resolve("project", (arr_t,), V, {"index": i}), ("arr",))
                  for i in range(n))
    nodes = [Node("addr", V, cands, region="core", depth=1)]
    constants = ()
    outputs = (("y", "addr"),)
    signals = (Signal("addr", "y", ("core",), V, "mse"),)
    if downstream == "eq":
        constants = (("c", Value.of(V, float(eq_constant))),)
        eq_op = registry.resolve("eq", (V, V), BOOL)
        nodes.append(Node("hit", BOOL, (Candidate(eq_op, ("addr", "c")),), region="core", depth=2))
        outputs = (("y", "hit"),)
        signals = (Signal("hit", "y", ("core",), BOOL, "mse"),)
    p = Program(inputs=(("arr", arr_t),), nodes=tuple(nodes), outputs=outputs,
                constants=constants, input_depths=(("arr", 0),))
    return p.validate(registry), signals


def index_program(registry, n, downstream="direct", eq_constant=0.0, init_addr=0.0):
    """One node `index(arr, addr)` with `addr` a trainable constant."""
    arr_t = product(*([V] * n))
    op = registry.resolve("index", (arr_t, ADDR), V)
    nodes = [Node("addr", V, (Candidate(op, ("arr", "b")),), region="core", depth=1)]
    constants = [("b", Value.of(ADDR, int(round(init_addr))))]
    outputs = (("y", "addr"),)
    signals = (Signal("addr", "y", ("core",), V, "mse"),)
    trainable = ("b",)
    if downstream == "eq":
        constants.append(("c", Value.of(V, float(eq_constant))))
        eq_op = registry.resolve("eq", (V, V), BOOL)
        nodes.append(Node("hit", BOOL, (Candidate(eq_op, ("addr", "c")),), region="core", depth=2))
        outputs = (("y", "hit"),)
        signals = (Signal("hit", "y", ("core",), BOOL, "mse"),)
        trainable = ("b", "c")
    p = Program(inputs=(("arr", arr_t),), nodes=tuple(nodes), outputs=outputs,
                constants=tuple(constants), state=(), input_depths=(("arr", 0),),
                trainable_constants=trainable)
    return p.validate(registry), signals


# --------------------------------------------------------------------------
# batched tensors
# --------------------------------------------------------------------------

def batch(program, examples, signals):
    inputs = {k: torch.stack([tensor(e["inputs"][k]) for e in examples]) for k, _ in program.inputs}
    targets = {s.target: torch.stack([tensor(e["targets"][s.target]) for e in examples]) for s in signals}
    return inputs, targets


def loss_at(model, inputs, targets, signals):
    _, _, trace = model(inputs, return_trace=True)
    BUDGET.add(len(model.program.nodes))
    return model.probe_loss(trace, targets, signals)


# --------------------------------------------------------------------------
# closed form for the DIRECT discrete case
# --------------------------------------------------------------------------

def quadratic_form(arrays, k):
    """`L(p) = p'Cp - 2 (Ce_k)'p + C_kk` with `C = E[x x']`.

    With DIRECT supervision the relaxed loss of the discrete address node is
    exactly this quadratic in the mixture weights, so the landscape over the
    simplex is fully determined by the second-moment matrix of the data.
    """
    x = arrays.double()
    C = (x.T @ x) / x.shape[0]
    return C, C[:, k], C[k, k]


# --------------------------------------------------------------------------
# scoring rules
# --------------------------------------------------------------------------

def gradient_pick(model, inputs, targets, signals, node="addr"):
    """The candidate the first Adam step raises: argmin of the logit gradient."""
    for p in model.choices:
        if p.grad is not None:
            p.grad = None
    loss = loss_at(model, inputs, targets, signals)
    loss.backward()
    i = [n.name for n in model.program.nodes].index(node)
    g = model.choices[i].grad
    if g is None or float(g.abs().sum()) == 0.0:
        return None, None
    return int(g.argmin()), g.detach().clone()


def perturbation_pick(model, inputs, targets, signals, node="addr"):
    """Score every candidate by hard-selecting it; return the argmin.

    This is the perturbation measurement, known from the perturbation-selection
    track to beat argmax under-trained.  Cost is |candidates| forward passes.
    """
    i = [n.name for n in model.program.nodes].index(node)
    n_cand = len(model.program.nodes[i].candidates)
    scores = []
    saved = dict(model.trials)
    for c in range(n_cand):
        model.trials = dict(saved)
        model.trials[node] = c
        with torch.no_grad():
            scores.append(float(loss_at(model, inputs, targets, signals)))
    model.trials = saved
    return int(min(range(n_cand), key=lambda c: scores[c])), scores


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(obj, indent=1, default=str))
    print("wrote", p, flush=True)
    return p
