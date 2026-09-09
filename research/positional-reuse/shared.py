"""Shared construction helpers for the positional-reuse experiments.

Everything here is built from operators already in `tcn/operators.py`. Nothing in
this file adds an operator, a type, or a preprocessing step.

The pattern under test, `hold / pair / map`:

    held    = insert(empty_set_constant, wide_tuple_input)   # set[Wide], capacity 1
    records = pair(position_constant_set, held)              # set[(Index, Wide)]
    mapped  = map(records; module = m)                       # set[(Index, Out)]

`pair` is the cartesian product, so `records` holds exactly one record per
position, each carrying the whole observation. `map` applies one crystallized
module to every record; the module reads its own position out of the record and
indexes the observation there. Three caller nodes, independent of the number of
positions.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from dataclasses import replace

from tcn.graph import Candidate, Node, Program
from tcn.operators import Registry
from tcn.scaffold import positional_scaffold
from tcn.types import BOOL, Value, floating, integer, product, setof

F = floating()
IDX = integer(16, signed=False)


def wide(width, element=F):
    return product(*(element for _ in range(width)))


class Builder:
    """Small helper that keeps node types and depths straight while wiring by hand."""

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
        node = Node(name, ops[0].output,
                    tuple(Candidate(o, tuple(sources)) for o in ops),
                    "core", depth, 0 if len(ops) == 1 else None)
        self.nodes.append(node)
        self.types[name] = node.output
        self.depths[name] = depth
        return name

    def program(self, outputs, **kw):
        return Program(self.inputs, tuple(self.nodes), tuple(outputs), self.constants,
                       input_depths=tuple((k, 0) for k, _ in self.inputs), **kw).validate(self.r)


def window_module(registry, width, offsets, body, name_hint="", element=F):
    """`(Index, Wide) -> (Index, body(window))`, crystallized and reusable.

    `body(builder, reads)` wires the shared computation from the window reads and
    returns the name of its single output node.
    """
    W = wide(width, element)
    REC = product(IDX, W)
    consts = tuple((f"off{k}", Value.of(IDX, off)) for k, off in enumerate(offsets) if off)
    b = Builder(registry, (("rec", REC),), consts)
    b.add("pos", "project", ["rec"], params={"index": 0})
    b.add("obs", "project", ["rec"], params={"index": 1})
    reads = []
    for k, off in enumerate(offsets):
        src = "pos" if not off else b.add(f"shift{k}", "add", ["pos", f"off{k}"])
        reads.append(b.add(f"read{k}", "index", ["obs", src]))
    out = body(b, reads, "")
    b.add("record", "tuple", ["pos", out])
    return b.program((("y", "record"),))


def shared_map_program(registry, width, positions, module_names, element=F):
    """hold / pair / map: three caller nodes for any number of positions.

    This is `tcn.scaffold.positional_scaffold`; the wrapper only keeps the output
    port named `y` so the measurement scripts read the same key everywhere.
    """
    prog = positional_scaffold(registry, wide(width, element), positions, module_names, index=IDX)
    return replace(prog, outputs=(("y", "mapped"),)).validate(registry)


def per_position_program(registry, width, positions, module_name, element=F):
    """The position-by-position equivalent: one call site per position.

    Same module, same results, but the caller names every position separately.
    """
    W = wide(width, element)
    consts = tuple((f"p{i}", Value.of(IDX, i)) for i in positions)
    b = Builder(registry, (("observation", W),), consts)
    calls = []
    for i in positions:
        rec = b.add(f"rec{i}", "tuple", [f"p{i}", "observation"])
        calls.append(b.add(f"call{i}", module_name, [rec]))
    b.add("all", "tuple", calls)
    return b.program((("y", "all"),))


def inlined_program(registry, width, positions, offsets, body, element=F):
    """No module at all: the shared computation written out at every position."""
    W = wide(width, element)
    consts = tuple((f"c{i}_{k}", Value.of(IDX, i + off))
                   for i in positions for k, off in enumerate(offsets))
    b = Builder(registry, (("observation", W),), consts)
    outs = []
    for i in positions:
        reads = [b.add(f"read{i}_{k}", "index", ["observation", f"c{i}_{k}"])
                 for k, _ in enumerate(offsets)]
        outs.append(body(b, reads, f"_{i}"))
    b.add("all", "tuple", outs)
    return b.program((("y", "all"),))
