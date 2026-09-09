"""Shared scaffolding for the code-generation track.

Target language: gate lists for `generators/logic` at width 3 -- a sequence of
`(wire_a, wire_b, table)` triples, run by that generator's own `evaluate`, and
expressible as the depth-independent `set[(index, wire_a, wire_b, table)]`
relation the generator already emits.

Specification: the behaviour, i.e. the 8-entry truth table of the target
function, as `tuple[bool x 8]`.  Nothing about the program is given.

Declared priors (AGENTS.md: state it, ablate it, certify it):

* The target *shape* is the Shannon skeleton of `bound.py` -- five gates, of
  which three are fixed wiring/tables and two carry the free fields `t0`, `t1`.
  `bound.py` certifies this shape is a bijection onto all 256 behaviours, and
  that free gate lists are massively non-unique, which is why a shape is
  declared at all.  The shape is a prior; the *contents* of both variable gates,
  and the ISA conventions needed to fill them, are searched.
* The stage-A scaffold declares the executor's node skeleton (a two-bit address,
  two tuple reads, one mux) but not which inputs form the address, in which
  order, which cofactor each read serves, or which input selects.  The stage-B
  scaffold declares eight address nodes and not one of their addresses.
* Stage A's supervision comes from the target ISA (`evaluate`), never from a
  hand-written executor.  Stage B's supervision is the behaviour alone: no
  reference netlist is ever shown to stage B.
"""
from __future__ import annotations
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tcn.graph import Program, Node, Candidate, Signal
from tcn.operators import Registry
from tcn.types import BOOL, Value, product, integer
from generators.logic.generator import evaluate, gate_set_value, gates_from_set

from bound import canonical, netlist_mask

WIDTH = 3
SPEC = product(*(BOOL,) * 8)          # the behaviour: f(m) for m = x0 + 2 x1 + 4 x2
NIB = product(*(BOOL,) * 4)           # one gate table, as four bits
ADDR = integer(3, signed=False)       # an address into the specification
TWO = integer(2, signed=False)        # a gate-table address
TABLE = integer(4, signed=False)      # a gate table, as the ISA writes it


class Builder:
    """Keeps node types and depths straight while wiring by hand."""

    def __init__(self, registry, inputs=(), constants=()):
        self.r = registry
        self.nodes = []
        self.types = dict(inputs) | {k: v.type for k, v in constants}
        self.depths = {k: 0 for k, _ in inputs} | {k: -1 for k, _ in constants}
        self.inputs = tuple(inputs)
        self.constants = tuple(constants)

    def add(self, name, op, sources, out=None, params=None):
        return self.choice(name, [(op, tuple(sources), out, params)])

    def choice(self, name, candidates, region="core"):
        cands = []
        for entry in candidates:
            op, sources, out, params = (tuple(entry) + (None, None))[:4]
            cands.append(Candidate(self.r.resolve(op, tuple(self.types[s] for s in sources),
                                                  out, params), tuple(sources)))
        outs = {c.operator.output for c in cands}
        if len(outs) != 1:
            raise TypeError(f"{name}: candidates disagree on output type")
        depth = max((self.depths[s] for c in cands for s in c.sources), default=0) + 1
        self.nodes.append(Node(name, cands[0].operator.output, tuple(cands), region, depth, None))
        self.types[name] = cands[0].operator.output
        self.depths[name] = depth
        return name

    def program(self, outputs):
        return Program(self.inputs, tuple(self.nodes), tuple(outputs), self.constants,
                       input_depths=tuple((k, 0) for k, _ in self.inputs)).validate(self.r)


# --- the target language ---------------------------------------------------
def behaviour(t0, t1):
    """The 8-bit behaviour of the canonical netlist, via the generator's evaluator."""
    gates = canonical(t0, t1)
    return tuple(evaluate(WIDTH, gates, m) for m in range(8))

def nib(t):
    return tuple(bool((t >> i) & 1) for i in range(4))

def artifact(t0, t1):
    """The emitted artifact: a legal `generators/logic` gate list and its relation view."""
    gates = canonical(t0, t1)
    rel = gate_set_value(gates, 5)
    assert gates_from_set(rel) == gates
    return gates, rel

def runs_correctly(spec, t0, t1):
    """Exact check: execute the emitted gate list on every assignment."""
    gates = canonical(t0, t1)
    return all(evaluate(WIDTH, gates, m) == spec[m] for m in range(8))


# --- specification corpus --------------------------------------------------
ALL_SPECS = tuple(tuple(bool((f >> m) & 1) for m in range(8)) for f in range(256))

def spec_value(spec):
    return Value.of(SPEC, tuple(spec))
