"""Rung 1 -- `signal`: recover frequency, then predict `future`, from raw samples.

Agent input is ONLY `observations['samples']` (the raw sample window).
`latent_states['frequencies']` and `probes['future']` are used as targets only.

The exact reference program the operator library admits (dt = 1):
    c      = (x[t] + x[t-2]) / (2 * x[t-1])        =  cos(2*pi*f)
    f      = atan2(sqrt(1 - c^2), c) / (2*pi)      =  acos(c)/(2*pi)
    x[t+1] = 2*c*x[t] - x[t-1]                     (iterate for t+2 .. t+4)
Both follow from the linear recurrence of a single sinusoid.  Nothing outside
`tcn/operators.py` is used, and no privileged value is ever an input.
"""
from __future__ import annotations
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tcn.generation import SCALAR, Value
from tcn.graph import Program, Node, Candidate, Signal
from tcn.operators import Registry
from tcn.types import product
from common import cand, episode_observation

F = SCALAR
WINDOW = 4
TICKS = 8
CFG = {"components": 1, "window": WINDOW, "horizon": 32}
S_TYPE = product(*(F for _ in range(WINDOW)))
T3 = product(F, F, F)

# node name -> (candidate menu, sources, region, index of the reference choice)
PICK = {"a": WINDOW - 1, "b": WINDOW - 2, "d": WINDOW - 3}
TRIG_OM = (("atan2", ("s", "c")), ("div", ("s", "c")), ("atan2", ("c", "s")), ("div", ("c", "s")))
MENUS = {
    "num": (("add", "sub", "mul", "min", "max"), (F, F), ("a", "d"), "algebra", 2),
    "den": (("mul", "add", "sub"), (F, F), ("b", "two"), "algebra", 2),
    "c":   (("div", "mul", "sub"), (F, F), ("num", "den"), "algebra", 3),
    "csq": (("mul", "add"), (F, F), ("c", "c"), "trig", 4),
    "t1":  (("sub", "add"), (F, F), ("one", "csq"), "trig", 5),
    "t2":  (("abs", "identity", "neg"), (F,), ("t1",), "trig", 6),
    "s":   (("sqrt", "identity", "abs"), (F,), ("t2",), "trig", 7),
    "fr":  (("div", "mul"), (F, F), ("om", "tau"), "latent", 9),
}
ALL_FREE = ("a", "b", "d", "num", "den", "c", "csq", "t1", "t2", "s", "om", "fr")


def sample_examples(seeds, min_abs=.05):
    """One observation per episode.  Only `samples` becomes a program input."""
    out = []
    for seed in seeds:
        obs, lat, pro = episode_observation("signal", seed, TICKS, CFG)
        xs = obs["samples"].decoded
        if min(abs(x) for x in xs) < min_abs:   # observable filter; no privileged information
            continue
        out.append({"inputs": {"samples": obs["samples"]},
                    "targets": {"frequency": Value.of(F, lat["frequencies"].decoded[0]),
                                "future": pro["future"]}})
    return out


def _node(r, name, free):
    if name in PICK:
        rng = range(WINDOW) if name in free else (PICK[name],)
        return Node(name, F, tuple(Candidate(r.resolve("project", (S_TYPE,), F, {"index": i}), ("samples",))
                                   for i in rng), "pick", 1)
    if name == "om":
        menu = TRIG_OM if "om" in free else TRIG_OM[:1]
        return Node("om", F, tuple(Candidate(r.resolve(op, (F, F)), src) for op, src in menu), "trig", 8)
    ops, types, sources, region, depth = MENUS[name]
    ops = ops if name in free else ops[:1]
    return Node(name, F, cand(r, ops, types, sources), region, depth)


def frequency_program(r, free=ALL_FREE, tau_trainable=False):
    """Nodes named in `free` keep their full menu; the rest are pinned to the reference."""
    order = ("a", "b", "d", "num", "den", "c", "csq", "t1", "t2", "s", "om", "fr")
    nodes = tuple(_node(r, n, set(free)) for n in order)
    constants = (("one", Value.of(F, 1.)), ("two", Value.of(F, 2.)),
                 ("tau", Value.of(F, 1. if tau_trainable else 2 * math.pi)))
    return Program((("samples", S_TYPE),), nodes, (("frequency", "fr"),), constants,
                   trainable_constants=("tau",) if tau_trainable else ())


def future_program(r, free=ALL_FREE, with_frequency_branch=True, free_recurrence=True):
    """The prefix a,b,d,num,den,c is SHARED by the future chain and the frequency branch,
    so a probe on `fr` supervises nodes that the `future` output also depends on."""
    order = ["a", "b", "d", "num", "den", "c"]
    if with_frequency_branch:
        order += ["csq", "t1", "t2", "s", "om", "fr"]
    nodes = [_node(r, n, set(free)) for n in order]
    prev2, prev = "b", "a"
    two_way = lambda ops: ops if free_recurrence else ops[:1]
    for step in range(4):
        m, m2, nx = f"m{step}", f"m2{step}", f"x{step}"
        base = 4 + 3 * step
        nodes += [
            Node(m, F, cand(r, two_way(("mul", "add")), (F, F), ("c", prev)), "recur", base),
            Node(m2, F, cand(r, two_way(("mul", "add")), (F, F), (m, "two")), "recur", base + 1),
            Node(nx, F, cand(r, two_way(("sub", "add")), (F, F), (m2, prev2)), "recur", base + 2),
        ]
        prev2, prev = prev, nx
    combos = (("x0", "x1", "x3"), ("x0", "x1", "x2"), ("x1", "x2", "x3"), ("x0", "x2", "x3"))
    combos = combos if free_recurrence else combos[:1]
    nodes.append(Node("future", T3, tuple(Candidate(r.resolve("tuple", (F, F, F)), srcs) for srcs in combos),
                      "readout", 16))
    constants = (("one", Value.of(F, 1.)), ("two", Value.of(F, 2.)), ("tau", Value.of(F, 2 * math.pi)))
    return Program((("samples", S_TYPE),), tuple(nodes), (("future", "future"),), constants)


FREQ_SIGNAL = Signal("fr", "frequency", ("latent",), F, "mse")
FUTURE_SIGNAL = Signal("future", "future", ("readout",), T3, "mse")

# graded scaffolds: progressively more of the reference program is left to search
LADDER = {
    "L0_om":        ("om",),
    "L1_om_s":      ("om", "s"),
    "L1b_+t2":      ("om", "s", "t2"),
    "L1c_+t1":      ("om", "s", "t2", "t1"),
    "L1d_+csq":     ("om", "s", "t2", "t1", "csq"),
    "L2_trig":      ("csq", "t1", "t2", "s", "om", "fr"),
    "L2b_+c":       ("csq", "t1", "t2", "s", "om", "fr", "c"),
    "L3_trig_alg":  ("num", "den", "c", "csq", "t1", "t2", "s", "om", "fr"),
    "L4_all":       ALL_FREE,
}
