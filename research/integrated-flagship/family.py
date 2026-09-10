"""The program family of PREREGISTRATION §3: slots, per-arm pools, the `tcn` Program.

One coarse graph (hand-initialisation H1) serves every arm; arms differ only in
the candidate pool of each choice slot. Pools are built through the repository's
own constructors wherever one exists -- `tcn.graph.legal_candidates` for the
address slots, `rung3_widgets.offset_pool` for the step schema -- so the
candidate *order* the fast engine uses is read off the `tcn` candidates, never
re-derived by hand.

Slot kinds: ADDR (address into the text), LIT (byte literal), GROUND (target
colour), MATCH (per-pixel colour test, a registered module), STEP (the click).
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE), str(ROOT / "research" / "visual-ladder")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                                   # noqa: E402
from tcn.generation import image_value, text_value                   # noqa: E402
from tcn.graph import Candidate, Node, Program, Signal, legal_candidates  # noqa: E402
from tcn.operators import Registry                                   # noqa: E402
from tcn.types import BOOL, Value, floating, integer, product, setof  # noqa: E402
from env import N_PIXELS, PALETTE, RESOLUTION, TEXT_CAPACITY         # noqa: E402
import rung3_widgets                                                 # noqa: E402

W = H = RESOLUTION
BYTE = integer(8, signed=False, role="byte")
TEXT_T = text_value("", TEXT_CAPACITY).type
LEN = TEXT_T.items[0]
TDATA = TEXT_T.items[1]
PIX_T = image_value(np.zeros((H, W, 3), dtype=np.uint8)).type
OBS = PIX_T.items[3]
IDX = integer(16, signed=False, overflow="wrap")
RGB = product(BYTE, BYTE, BYTE)
OT = product(OBS, RGB)
REC = product(IDX, OT)
KEY = integer(24, signed=False)
F = floating()

SLOTS = ("cpos", "lc1", "lc2", "lc3", "K1", "K2", "K3", "K4",
         "ra", "lr1", "lr2", "M", "X1", "X2", "X3")
KIND = {"cpos": "ADDR", "ra": "ADDR", "lc1": "LIT", "lc2": "LIT", "lc3": "LIT",
        "lr1": "LIT", "lr2": "LIT", "K1": "GROUND", "K2": "GROUND", "K3": "GROUND",
        "K4": "GROUND", "M": "MATCH", "X1": "STEP", "X2": "STEP", "X3": "STEP"}

TEXT_CONSTS = tuple(range(16))
ADDR_PORTS = {"length": LEN} | {f"k{i}": LEN for i in TEXT_CONSTS}
LETTERS = tuple(range(ord("a"), ord("z") + 1))
STEP_FLAT_OPS = ("add", "sub", "mul", "min", "max")
STEP_FLAT_OFFSETS = tuple(range(1, 129))
BASES = ("lo", "hi")


def offset_pool(width):
    """§33's step schema pool, imported, never retyped: (6, 3W+3, 3, 3W, 9)."""
    return tuple(rung3_widgets.offset_pool(width))


def distractor_offsets(width):
    """Same size and shape as `offset_pool`, wrong content (PREREGISTRATION §4.4)."""
    return (4, 3 * width + 2, 5, 3 * width - 1, 7)


# --------------------------------------------------------------------- pools
def addr_pool(registry, names, arities):
    return legal_candidates(registry, names, ADDR_PORTS, LEN, arities=arities)


def addr_spec(candidate):
    return (candidate.operator.name,) + tuple(candidate.sources)


def pools(arm, width=W):
    """Candidate specs per slot for an arm's pools: 'flat', 'schema' or 'distractor'."""
    r = Registry()
    if arm == "flat":
        addr = addr_pool(r, ("identity", "add", "sub", "mul", "min", "max"), (1, 2))
        match = [(sg, sb, rg, same) for sg in (1, 2, 3) for sb in (1, 2, 3)
                 for rg in range(16) for same in range(16)]
        step = [(op, b, o) for op in STEP_FLAT_OPS for b in BASES for o in STEP_FLAT_OFFSETS]
    elif arm == "schema":
        addr = addr_pool(r, ("add", "sub", "mul", "min", "max"), (2,))
        match = [(1, 2, rg, same) for rg in range(16) for same in range(16)]
        step = [(op, b, o) for op in ("add", "sub") for b in BASES for o in offset_pool(width)]
    elif arm == "distractor":
        addr = addr_pool(r, ("mul", "min", "max", "idiv", "mod"), (2,))
        match = [(2, 1, rg, same) for rg in range(16) for same in range(16)]
        step = [(op, b, o) for op in ("add", "sub") for b in BASES
                for o in distractor_offsets(width)]
    else:
        raise ValueError(arm)
    addr = [addr_spec(c) for c in addr]
    lit = list(LETTERS)
    ground = list(range(len(PALETTE)))
    out = {}
    for slot in SLOTS:
        out[slot] = {"ADDR": addr, "LIT": lit, "GROUND": ground, "MATCH": match,
                     "STEP": step}[KIND[slot]]
    return {k: list(v) for k, v in out.items()}


# --------------------------------------------------------------- tcn program
class _B:
    """Keeps node types and depths straight while wiring (as `visual-ladder`'s Builder)."""

    def __init__(self, registry, inputs, constants):
        self.r = registry
        self.nodes = []
        self.types = dict(inputs) | {k: v.type for k, v in constants}
        self.depths = {k: 0 for k, _ in inputs} | {k: -1 for k, _ in constants}
        self.inputs = tuple(inputs)
        self.constants = list(constants)

    def const(self, name, t, value):
        if name not in self.types:
            v = Value.of(t, value) if not isinstance(value, Value) else value
            self.constants.append((name, v))
            self.types[name] = v.type
            self.depths[name] = -1
        return name

    def node(self, name, candidates):
        """`candidates`: list of (operator, sources, output, params)."""
        cands = []
        for op, sources, out, params in candidates:
            cands.append(Candidate(self.r.resolve(op, tuple(self.types[s] for s in sources),
                                                  out, params), tuple(sources)))
        if len({c.operator.output for c in cands}) != 1:
            raise TypeError(f"{name}: candidates disagree on output type")
        depth = max(self.depths[s] for c in cands for s in c.sources) + 1
        self.nodes.append(Node(name, cands[0].operator.output, tuple(cands), "core", depth, None))
        self.types[name] = cands[0].operator.output
        self.depths[name] = depth
        return name

    def one(self, name, op, sources, out=None, params=None):
        return self.node(name, [(op, sources, out, params)])

    def program(self, outputs):
        return Program(self.inputs, tuple(self.nodes), tuple(outputs), tuple(self.constants),
                       input_depths=tuple((k, 0) for k, _ in self.inputs)).validate(self.r)


def matcher_module(registry, spec):
    """(s_g, s_b, rg, same) as a frozen module: does the pixel at p carry colour t?"""
    sg, sb, rg, same = spec
    b = _B(registry, (("rec", REC),), ())
    b.const("sg", IDX, sg)
    b.const("sb", IDX, sb)
    b.one("p", "project", ["rec"], params={"index": 0})
    b.one("inner", "project", ["rec"], params={"index": 1})
    b.one("obs", "project", ["inner"], params={"index": 0})
    b.one("t", "project", ["inner"], params={"index": 1})
    for i, ch in enumerate("rgb"):
        b.one(f"t{ch}", "project", ["t"], params={"index": i})
    b.one("pg", "add", ["p", "sg"])
    b.one("pb", "add", ["p", "sb"])
    b.one("vr", "index", ["obs", "p"])
    b.one("vg", "index", ["obs", "pg"])
    b.one("vb", "index", ["obs", "pb"])
    b.one("er", "eq", ["vr", "tr"])
    b.one("eg", "eq", ["vg", "tg"])
    b.one("eb", "eq", ["vb", "tb"])
    b.one("rg", f"truth_{rg}", ["er", "eg"])
    b.one("same", f"truth_{same}", ["rg", "eb"])
    p = b.program((("y", "same"),))
    return registry.register_module(p.harden({n.name: 0 for n in p.nodes}))


def projection_module(registry):
    b = _B(registry, (("rec", REC),), ())
    b.one("p", "project", ["rec"], params={"index": 0})
    p = b.program((("y", "p"),))
    return registry.register_module(p.harden({n.name: 0 for n in p.nodes}))


def build(pool, registry=None, head="agent"):
    """The whole scaffold for the given per-slot pools.

    `head='agent'` appends the declared encoder and policy that `tcn.agent.Agent`
    needs; `head='key'` appends the scoring head used by the `tcn.search.enumerate_*`
    equivalence check (PREREGISTRATION §6.3 V2): the packed colour of the clicked
    pixel, which equals the target's colour exactly when the click hits it.
    """
    r = registry or Registry()
    inputs = [("text", TEXT_T), ("pixels", PIX_T)]
    if head == "agent":
        from env import CLICK
        inputs.append(("action.0.pos", CLICK))
    b = _B(r, inputs, ())
    for i in TEXT_CONSTS:
        b.const(f"k{i}", LEN, i)
    b.one("length", "project", ["text"], params={"index": 0})
    b.one("tdata", "project", ["text"], params={"index": 1})
    b.one("obs", "project", ["pixels"], params={"index": 3})

    def addr(slot):
        return b.node(slot, [(spec[0], list(spec[1:]), None, None) for spec in pool[slot]])

    def lit(slot, byte):
        cands = []
        for v in pool[slot]:
            cands.append(("eq", [byte, b.const(f"b{v}", BYTE, v)], None, None))
        return b.node(slot, cands)

    addr("cpos")
    b.one("cbyte", "index", ["tdata", "cpos"])
    for s in ("lc1", "lc2", "lc3"):
        lit(s, "cbyte")
    for s in ("K1", "K2", "K3", "K4"):
        b.node(s, [("identity", [b.const(f"col{i}", RGB, tuple(PALETTE[i]))], None, None)
                   for i in pool[s]])
    b.one("t3", "mux", ["lc3", "K3", "K4"])
    b.one("t2", "mux", ["lc2", "K2", "t3"])
    b.one("t", "mux", ["lc1", "K1", "t2"])
    b.one("ot", "tuple", ["obs", "t"])
    b.const("empty", setof(OT, 1), ())
    b.const("positions", setof(IDX, N_PIXELS), tuple(3 * i for i in range(N_PIXELS)))
    b.one("held", "insert", ["empty", "ot"])
    b.one("records", "pair", ["positions", "held"])
    modules = [matcher_module(r, spec) for spec in pool["M"]]
    b.node("M", [("filter", ["records"], None, {"module": m}) for m in modules])
    b.one("anchor", "map", ["M"], params={"module": projection_module(r)})
    b.one("lo", "reduce_min", ["anchor"])
    b.one("hi", "reduce_max", ["anchor"])
    addr("ra")
    b.one("rbyte", "index", ["tdata", "ra"])
    for s in ("lr1", "lr2"):
        lit(s, "rbyte")
    for s in ("X1", "X2", "X3"):
        b.node(s, [(op, [base, b.const(f"o{off}", IDX, off)], None, None)
                   for op, base, off in pool[s]])
    b.one("c2", "mux", ["lr2", "X2", "X3"])
    b.one("clk", "mux", ["lr1", "X1", "c2"])
    b.const("three", IDX, 3)
    b.one("q", "idiv", ["clk", "three"])
    outputs = [("pos", "q")]
    if head == "agent":
        b.const("one_f", F, 1.)
        b.const("n_f", F, float(N_PIXELS))
        b.const("half_f", F, .5)
        b.const("logstd", F, -5.)
        b.const("logit", product(F), (1.,))
        b.one("qf", "decode", ["q"], out=F)
        b.one("num", "add", ["qf", "one_f"])
        b.one("den", "sub", ["n_f", "qf"])
        b.one("ratio", "div", ["num", "den"])
        b.one("lg", "log", ["ratio"])
        b.one("mean", "mul", ["lg", "half_f"])
        b.one("arg", "tuple", ["mean", "logstd"])
        b.one("policy", "identity", ["logit"])
        outputs = [("policy", "policy"), ("arg", "arg")] + outputs
    elif head == "key":
        b.const("one_i", IDX, 1)
        b.const("two_i", IDX, 2)
        b.one("p0", "mul", ["q", "three"])
        b.one("p1", "add", ["p0", "one_i"])
        b.one("p2", "add", ["p0", "two_i"])
        b.one("v0", "index", ["obs", "p0"])
        b.one("v1", "index", ["obs", "p1"])
        b.one("v2", "index", ["obs", "p2"])
        b.one("kt", "tuple", ["v0", "v1", "v2"])
        b.one("key", "pack", ["kt"], out=KEY)
        outputs = [("key", "key")] + outputs
    return b.program(outputs), r


def key_signal():
    return (Signal("key", "key", ("core",), KEY, "mse"),)


def pack_rgb(rgb):
    r, g, bl = rgb
    return int(r) | (int(g) << 8) | (int(bl) << 16)


def full_selection(program, chosen):
    """Slot indices plus index 0 on every single-candidate plumbing node."""
    out = {n.name: 0 for n in program.nodes}
    out.update(chosen)
    return out
