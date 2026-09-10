"""A depth-parametric scaffold over the width-varying `program` observation.

The point of this file is that it is a **schema**, not an artifact. Calling it at
depth `d` returns a `Program` exactly typed for that depth's `program`
observation -- `tuple[3*d]` -- and validated by `Program.validate` with nothing
relaxed. What is shared across depths is the *shape of the choice*: exactly two
nodes carry candidates, with counts (17, 16) at every depth, so an integer
selection vector found at one width names a program at every other width.

Whether that vector is still *correct* at another width is the measurement, not
an assumption. Nothing here weakens a type check; the audit in `audit.py` lists
every check that stays in force.

The policy tail, its five constants and the action schema are copied verbatim
from `examples/joint.py` by way of
`research/depth-generalization/interpreter.py`, so returns are on the same 0-4
scale as FINDINGS section 15.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tcn.generation import Host, Action
from tcn.graph import Program, Node, Candidate
from tcn.operators import Registry
from tcn.search import AgentConfig
from tcn.types import BOOL, Value, floating, integer, product

F = floating()
IDX = integer(8, signed=False)

OBJECTIVES = ({"invert": False}, {"invert": True})

CONSTANTS = (("w0", Value.of(F, -2.)), ("w1", Value.of(F, 2.)),
             ("bias0", Value.of(F, 1.)), ("bias1", Value.of(F, -1.)),
             ("baseline", Value.of(F, 1.)))

ACTIONS = (Action("answer", arguments=(("value", Value.of(BOOL, False)),)),
           Action("answer", arguments=(("value", Value.of(BOOL, True)),)))

OUTPUTS = (("policy", "policy"), ("prediction", "prediction"),
           ("probe", "prediction"), ("value", "value"))


class _Builder:
    """Nodes plus a depth ledger, so every edge respects the depth scaffold."""

    def __init__(self, registry, inputs):
        self.r = registry
        self.nodes = []
        self.types = dict(inputs)
        self.depth = {k: 0 for k, _ in inputs}
        for k, v in CONSTANTS:
            self.types[k] = v.type
            self.depth[k] = -1

    def add(self, name, candidates, region="core"):
        out = candidates[0].operator.output
        d = max((self.depth[s] for c in candidates for s in c.sources), default=0) + 1
        self.nodes.append(Node(name, out, tuple(candidates), region, max(d, 1)))
        self.types[name] = out
        self.depth[name] = max(d, 1)
        return name

    def op(self, name, sources, output=None, parameters=None):
        return Candidate(self.r.resolve(name, tuple(self.types[s] for s in sources),
                                        output, parameters), tuple(sources))


def _policy_tail(b, relation="relation", goal="goal_relation"):
    """The z/world/prediction/policy/value tail of examples/joint.py, verbatim."""
    b.add("z", [b.op("encode", (goal,), F)], "encoding")
    b.add("world", [b.op("encode", (relation,), F)], "encoding")
    for i in range(2):
        b.add(f"mul{i}", [b.op("mul", ("z", f"w{i}"))], "policy")
        b.add(f"logit{i}", [b.op("add", (f"mul{i}", f"bias{i}"))], "policy")
    b.add("policy", [b.op("tuple", ("logit0", "logit1"))], "policy")
    b.add("prediction", [b.op("tuple", ("z", "world"))], "prediction")
    b.add("value", [b.op("tuple", ("baseline",))], "value")


def interpreter_scaffold(host, depth, registry=None, answer="last"):
    """One instantiated artifact per depth, from one schema.

    Reads the `program` channel directly -- the width-varying one -- and unrolls
    the circuit: gate `i` reads fields `3i, 3i+1, 3i+2`, looks its two operands
    up in the wire tuple built so far with `index`, and selects its truth table
    with a second `index` over the sixteen possible gate outputs.

    Choice lives at exactly two nodes and nowhere else:

    * `relation`  -- 17 candidates: the sixteen fixed `truth_j` over the last
      gate's operands, plus `identity(gate_{d-1})`, the table-conditioned answer.
    * `goal_relation` -- 16 candidates: `truth_j(relation, goal)`.
    """
    r = registry or Registry()
    view = host.view().observations
    names = ("bits", "goal", "program")
    inputs = tuple((k, view[k].type) for k in names) + (("action", product(F, F)), ("dt", F))
    ptype, btype = dict(inputs)["program"], dict(inputs)["bits"]
    width = len(btype.items)
    if len(ptype.items) != 3 * depth:
        raise ValueError(f"program observation has {len(ptype.items)} fields, expected {3 * depth}")

    b = _Builder(r, inputs)
    for j in range(width):
        b.add(f"bit_{j}", [b.op("project", ("bits",), parameters={"index": j})], "observation")

    wires = [f"bit_{j}" for j in range(width)]
    for i in range(depth):
        for label, offset in (("a", 0), ("b", 1), ("t", 2)):
            b.add(f"p{label}_{i}", [b.op("project", ("program",),
                                         parameters={"index": 3 * i + offset})], "observation")
            b.add(f"i{label}_{i}", [b.op("encode", (f"p{label}_{i}",), IDX)], "observation")
        b.add(f"wires_{i}", [b.op("tuple", tuple(wires))], "latent")
        b.add(f"wa_{i}", [b.op("index", (f"wires_{i}", f"ia_{i}"))], "latent")
        b.add(f"wb_{i}", [b.op("index", (f"wires_{i}", f"ib_{i}"))], "latent")
        for j in range(16):
            b.add(f"g{j}_{i}", [b.op(f"truth_{j}", (f"wa_{i}", f"wb_{i}"))], "latent")
        b.add(f"tab_{i}", [b.op("tuple", tuple(f"g{j}_{i}" for j in range(16)))], "latent")
        b.add(f"gate_{i}", [b.op("index", (f"tab_{i}", f"it_{i}"))], "latent")
        wires = wires + [f"gate_{i}"]

    last = depth - 1
    fixed = [b.op(f"truth_{j}", (f"wa_{last}", f"wb_{last}")) for j in range(16)]
    # `answer="first"` is amendment 1's wrong-schema control: the lookup reads
    # gate 0 rather than the last gate. At depth 1 the two are the same node, so
    # a depth-1 fit cannot tell the two schemas apart.
    lookup = b.op("identity", (f"gate_{0 if answer == 'first' else last}",))
    b.add("relation", fixed + [lookup], "latent")
    b.add("goal_relation", [b.op(f"truth_{j}", ("relation", "goal")) for j in range(16)], "latent")
    _policy_tail(b)

    program = Program(inputs, tuple(b.nodes), OUTPUTS, CONSTANTS,
                      trainable_constants=tuple(k for k, _ in CONSTANTS)).validate(r)
    return names, program, r


def record_scaffold(host, depth=None, registry=None):
    """Negative control: `bits` and `goal` only, no `program` port at all.

    Its selection vector is width-free trivially -- the scaffold does not read
    the width-varying channel -- so it separates "the vector is applicable at
    another width" from "the capability survives at another width".
    Counts are (16, 16); the graph is byte-identical at every depth.
    """
    r = registry or Registry()
    view = host.view().observations
    names = ("bits", "goal")
    inputs = tuple((k, view[k].type) for k in names) + (("action", product(F, F)), ("dt", F))
    b = _Builder(r, inputs)
    for j in range(2):
        b.add(f"bit_{j}", [b.op("project", ("bits",), parameters={"index": j})], "observation")
    b.add("relation", [b.op(f"truth_{j}", ("bit_0", "bit_1")) for j in range(16)], "latent")
    b.add("goal_relation", [b.op(f"truth_{j}", ("relation", "goal")) for j in range(16)], "latent")
    _policy_tail(b)
    program = Program(inputs, tuple(b.nodes), OUTPUTS, CONSTANTS,
                      trainable_constants=tuple(k for k, _ in CONSTANTS)).validate(r)
    return names, program, r


def first_gate_scaffold(host, depth, registry=None):
    """Amendment 1, arm F. Identical to `interpreter_scaffold` but for one edge."""
    return interpreter_scaffold(host, depth, registry, answer="first")


SCAFFOLDS = {"interpreter": interpreter_scaffold, "record": record_scaffold,
             "first_gate": first_gate_scaffold}


def agent_config(names, horizon):
    return AgentConfig(tuple(names), ACTIONS, (), 1., horizon)


def make_host(depth, index, split, base, horizon):
    return Host.create("logic", seed=0, index=index, split=split,
                       configuration=dict(base) | {"depth": depth, "horizon": horizon},
                       objective=OBJECTIVES[index % len(OBJECTIVES)])
