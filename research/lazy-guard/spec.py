"""The two miniature specifications, built as real frozen `tcn.graph.Program`s.

Nothing under `tcn/` is modified.  Each spec is written in today's algebra
exactly as `research/algorithm-resynthesis/DESIGN.md` sec 1 says it must be: a
total, eager, first-order dataflow DAG.  `Program.execute` assigns every node
once per tick, so the expensive continuation is evaluated at every position
whether or not the cheap predicate holds.  That is the fact the experiment
attacks; it is a property of the language, not of these programs.

Two miniatures:

  * `sparse_guard`  -- N positions, cheap predicate p_i, expensive value e_i,
                       output {(i, e_i) : p_i}.  Domain 16^4 = 65,536.
  * `find_first`    -- N positions, an unrolled family of predicate modules,
                       output the first index that satisfies one, else N.
                       Domain 4^8 = 65,536.

Both domains are exhausted, so every number reported about them is exact rather
than sampled.
"""
from __future__ import annotations

import itertools
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tcn.graph import Candidate, Node, Program
from tcn.operators import Registry
from tcn.types import BOOL, Type, Value, integer, product, setof

U2 = integer(2, signed=False)
U4 = integer(4, signed=False)
U8 = integer(8, signed=False)
U16 = integer(16, signed=False)


def _chain(reg, in_t, steps):
    """A straight-line single-input module.  `steps` is [(op, const_or_None, out_t)]."""
    nodes = []
    consts = []
    prev = "x"
    prev_t = in_t
    depth = 1
    for k, (op, arg, ot) in enumerate(steps):
        nm = "t%d" % k
        if arg is None:
            cand = Candidate(reg.resolve(op, (prev_t,), ot), (prev,))
        else:
            cval, ct = arg
            cname = "c%d" % k
            consts.append((cname, Value.of(ct, cval)))
            cand = Candidate(reg.resolve(op, (prev_t, ct), ot), (prev, cname))
        nodes.append(Node(nm, ot, (cand,), depth=depth, selected=0))
        prev, prev_t = nm, ot
        depth += 1
    p = Program(inputs=(("x", in_t),), nodes=tuple(nodes), outputs=(("y", prev),),
                constants=tuple(consts))
    p.validate(reg)
    return reg.register_module(p)


# --------------------------------------------------------------------------
# expensive value module E : U4 -> U16
# --------------------------------------------------------------------------
def expensive_module(reg, coeffs=((7, 5), (11, 3), (5, 9), (13, 1), (3, 7)), src=U4):
    steps = [("encode", None, U16)]
    for a, b in coeffs:
        steps.append(("mul", (a, U16), U16))
        steps.append(("add", (b, U16), U16))
        steps.append(("mod", (251, U16), U16))
    return _chain(reg, src, steps)


# --------------------------------------------------------------------------
# predicate module P : U2 -> BOOL, a real chain; truth rate is measured
# --------------------------------------------------------------------------
def predicate_module(reg, a=5, b=3, m=7, thr=4):
    steps = [("encode", None, U8),
             ("mul", (a, U8), U8),
             ("add", (b, U8), U8),
             ("mod", (m, U8), U8),
             ("mul", (a, U8), U8),
             ("mod", (m, U8), U8),
             ("ge", (thr, U8), BOOL)]
    return _chain(reg, U2, steps)


def module_truth_rate(reg, name, in_t, values):
    m = reg.modules[name]
    hits = 0
    for v in values:
        out, _ = m.run({"x": Value.of(in_t, v)}, registry=reg)
        if next(iter(out.values())).decoded:
            hits += 1
    return hits / len(values)


# --------------------------------------------------------------------------
# STAGE 1 spec: guard-dominated sparse evaluation
# --------------------------------------------------------------------------
def sparse_guard(n=4, hit=15, bits=4, coeffs=((7, 5), (11, 3), (5, 9), (13, 1), (3, 7))):
    reg = Registry()
    U = integer(bits, signed=False)
    E = expensive_module(reg, coeffs, U)
    xt = product(*([U] * n))
    el = product(U, U16)
    st = setof(el, n)
    consts = [("K", Value.of(U, hit)), ("EMPTY", Value.of(st, frozenset()))]
    for i in range(n):
        consts.append(("I%d" % i, Value.of(U, i)))
    nodes = []
    prev_set = "EMPTY"
    d = 1
    for i in range(n):
        nodes.append(Node("x%d" % i, U, (Candidate(reg.resolve("project", (xt,), U, {"index": i}), ("x",)),), depth=d, selected=0))
        nodes.append(Node("p%d" % i, BOOL, (Candidate(reg.resolve("eq", (U, U), BOOL), ("x%d" % i, "K")),), depth=d + 1, selected=0))
        nodes.append(Node("e%d" % i, U16, (Candidate(reg.resolve(E, (U,), U16), ("x%d" % i,)),), depth=d + 1, selected=0))
        nodes.append(Node("t%d" % i, el, (Candidate(reg.resolve("tuple", (U, U16), el), ("I%d" % i, "e%d" % i)),), depth=d + 2, selected=0))
        nodes.append(Node("j%d" % i, st, (Candidate(reg.resolve("insert", (st, el), st), (prev_set, "t%d" % i)),), depth=d + 3, selected=0))
        nodes.append(Node("s%d" % i, st, (Candidate(reg.resolve("mux", (BOOL, st, st), st), ("p%d" % i, "j%d" % i, prev_set)),), depth=d + 4, selected=0))
        prev_set = "s%d" % i
        d += 5
    p = Program(inputs=(("x", xt),), nodes=tuple(nodes), outputs=(("out", prev_set),),
                constants=tuple(consts))
    p.validate(reg)
    return {"name": "sparse_guard", "program": p, "registry": reg, "n": n,
            "input_type": xt, "carrier": U, "carrier_values": list(range(2 ** bits)),
            "expensive": E, "hit": hit, "port": "x"}


def sparse_guard_domain(fx):
    for combo in itertools.product(fx["carrier_values"], repeat=fx["n"]):
        yield {"x": combo}


# --------------------------------------------------------------------------
# STAGE 2 spec: an unrolled predicate family and a mux chain
# --------------------------------------------------------------------------
def find_first(n=8, a=5, b=3, m=7, thr=4):
    reg = Registry()
    P = predicate_module(reg, a, b, m, thr)
    yt = product(*([U2] * n))
    consts = [("N", Value.of(U4, n))]
    for i in range(n):
        consts.append(("I%d" % i, Value.of(U4, i)))
    nodes = []
    d = 1
    for i in range(n):
        nodes.append(Node("y%d" % i, U2, (Candidate(reg.resolve("project", (yt,), U2, {"index": i}), ("y",)),), depth=d, selected=0))
        nodes.append(Node("q%d" % i, BOOL, (Candidate(reg.resolve(P, (U2,), BOOL), ("y%d" % i,)),), depth=d + 1, selected=0))
        d += 2
    prev = "N"
    for i in reversed(range(n)):
        nm = "r%d" % i
        nodes.append(Node(nm, U4, (Candidate(reg.resolve("mux", (BOOL, U4, U4), U4), ("q%d" % i, "I%d" % i, prev)),), depth=d, selected=0))
        prev = nm
        d += 1
    p = Program(inputs=(("y", yt),), nodes=tuple(nodes), outputs=(("out", prev),),
                constants=tuple(consts))
    p.validate(reg)
    return {"name": "find_first", "program": p, "registry": reg, "n": n,
            "input_type": yt, "carrier": U2, "carrier_values": list(range(4)),
            "predicate": P, "port": "y"}


def find_first_domain(fx):
    for combo in itertools.product(fx["carrier_values"], repeat=fx["n"]):
        yield {"y": combo}


FIXTURES = {"sparse_guard": (sparse_guard, sparse_guard_domain),
            "find_first": (find_first, find_first_domain)}


if __name__ == "__main__":
    for nm, (build, dom) in FIXTURES.items():
        fx = build()
        p, r = fx["program"], fx["registry"]
        print(nm, "nodes", len(p.nodes), "modules", len(r.modules),
              "execution_cost", p.execution_cost(r), "digest", p.digest)
        port = fx["port"]
        for c in itertools.islice(dom(fx), 3):
            ins = {port: Value.of(fx["input_type"], c[port])}
            out, _ = p.run(ins, registry=r)
            print("   ", c, "->", {k: v.decoded for k, v in out.items()})
    fx = find_first()
    print("P truth rate", module_truth_rate(fx["registry"], fx["predicate"], U2, [0, 1, 2, 3]))
    fx = sparse_guard()
    print("E values", [fx["registry"].modules[fx["expensive"]].run({"x": Value.of(U4, v)}, registry=fx["registry"])[0]["y"].decoded for v in range(16)])
