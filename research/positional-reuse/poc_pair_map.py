"""Can one crystallized module be applied at N positions with O(1) nodes?

The idea under test: `pair` forms a cartesian product of two sets. Take a
constant set of position indices and a singleton set holding the wide tuple;
their pair is the set of (position, whole-input) records, one per position.
`map` then applies one module to every record. The module indexes the wide
tuple at its own position. Node count is independent of N.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tcn.types import BOOL, Value, integer, floating, product, setof
from tcn.operators import Registry
from tcn.graph import Program, Node, Candidate

F = floating()
IDX = integer(16, signed=False)


def wide_type(width):
    return product(*(F for _ in range(width)))


def make_module(width, registry, offsets=(0,)):
    """(position, wide tuple) -> (position, mean of the window at that position)."""
    WIDE = wide_type(width)
    REC = product(IDX, WIDE)
    r = registry
    nodes = []
    types = {"rec": REC}

    def add(name, op, sources, out=None, params=None, depth=None):
        o = r.resolve(op, tuple(types[s] for s in sources), out, params)
        n = Node(name, o.output, (Candidate(o, tuple(sources)),), "core", depth, 0)
        nodes.append(n)
        types[name] = o.output
        return name

    add("pos", "project", ["rec"], params={"index": 0}, depth=1)
    add("wide", "project", ["rec"], params={"index": 1}, depth=1)
    consts = []
    reads = []
    for k, off in enumerate(offsets):
        if off == 0:
            src = "pos"
        else:
            types[f"off{k}"] = IDX
            consts.append((f"off{k}", Value.of(IDX, off)))
            src = add(f"shift{k}", "add", ["pos", f"off{k}"], depth=2)
        reads.append(add(f"read{k}", "index", ["wide", src], depth=3))
    add("win", "tuple", reads, depth=4)
    add("val", "mean", ["win"], depth=5)
    add("out", "tuple", ["pos", "val"], depth=6)
    return Program((("rec", REC),), tuple(nodes), (("y", "out"),), tuple(consts),
                   input_depths=(("rec", 0),)).validate(r)


def positional_program(width, n, registry, offsets=(0,)):
    """3 structural nodes plus one module definition, independent of n."""
    r = registry
    WIDE = wide_type(width)
    mod = make_module(width, r, offsets)
    mname = r.register_module(mod)
    IDXSET = setof(IDX, n)
    ONE = setof(WIDE, 1)
    types = {"pixels": WIDE, "positions": IDXSET, "empty": ONE}
    nodes = []

    def add(name, op, sources, out=None, params=None, depth=None):
        o = r.resolve(op, tuple(types[s] for s in sources), out, params)
        node = Node(name, o.output, (Candidate(o, tuple(sources)),), "core", depth, 0)
        nodes.append(node)
        types[name] = o.output
        return name

    add("held", "insert", ["empty", "pixels"], depth=1)
    add("records", "pair", ["positions", "held"], depth=2)
    add("mapped", "map", ["records"], params={"module": mname}, depth=3)
    consts = (("positions", Value.of(IDXSET, tuple(range(n)))), ("empty", Value.of(ONE, ())))
    prog = Program((("pixels", WIDE),), tuple(nodes), (("y", "mapped"),), consts,
                   input_depths=(("pixels", 0),)).validate(r)
    return prog, mname, mod


if __name__ == "__main__":
    r = Registry()
    width = 6
    offsets = (0, 1)
    n = width - max(offsets)          # valid windows only; no clamping needed
    prog, mname, mod = positional_program(width, n, r, offsets=offsets)
    xs = tuple(float(i) * 1.5 for i in range(width))
    out, _ = prog.run({"pixels": Value.of(wide_type(width), xs)}, registry=r)
    print("input  ", xs)
    print("output ", sorted(out["y"].decoded))
    print("caller nodes:", len(prog.nodes), " module nodes:", len(mod.nodes))
