"""Cost instrumentation.  The precondition of DESIGN sec 0.3 and sec 11.

The objective in the repository today cannot see early exit.  `filter` is
charged at declared *capacity* (`tcn/operators.py:118`), not realized size, and
`Program.execution_cost` is a static sum over selected candidates, so it takes
the same value for every input.  Section 41 measured that directly: 148.0 across
ten programs whose executed bytecodes differed by 48.  Early exit never improves
the worst case, so a metric that only reports a worst case can never reward it.

This module reports three numbers, always separately, never collapsed:

  * `worst_ops`     -- max executed primitive operations over the declared
                       finite domain.  Exhaustive, so it is the true maximum,
                       not a bound.
  * `expected_ops`  -- mean executed primitive operations over the declared
                       input distribution.  Exhaustive over the domain, so it is
                       an exact expectation, not a sample mean.
  * `bytecodes`     -- executed CPython bytecodes, worst and expected, counted
                       with `sys.monitoring` exactly as
                       `research/compiled-runtime/attribute.py` counts them.

A *primitive operation* is one application of a leaf `tcn` operator.  A module
call is not itself a primitive; the leaves inside it are.  Both the
specification and the resynthesized algorithm are counted by the same
instrument, so the two numbers are commensurable by construction.

`NativeEval` is a generic mini-interpreter over *decoded* values.  It is not a
hand-written re-statement of either miniature: it walks whatever `Program` it is
given.  `certify_native` checks it against `Registry.exact` on **every** input of
the declared domain, so it is certified rather than assumed.
"""
from __future__ import annotations

import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tcn.types import Value, decode


class Meter:
    """Executed primitive operations, and a per-operator breakdown."""

    def __init__(self):
        self.n = 0
        self.by = {}

    def hit(self, name, k=1):
        self.n += k
        self.by[name] = self.by.get(name, 0) + k

    def reset(self):
        self.n = 0
        self.by = {}


def _canon_int(t, x):
    """Range-check an integer result exactly as `tcn.types.encode` would."""
    M = 2 ** t.bits
    lo = -(M // 2) if t.encoding.signed else 0
    hi = M // (2 if t.encoding.signed else 1) - 1
    if not lo <= x <= hi:
        raise OverflowError("value outside integer range")
    return x


class NativeEval:
    """Decoded-value evaluator over a frozen `Program`, with a primitive meter.

    Generic: it dispatches on operator name only.  Anything it has not been
    taught raises, so silent divergence from `Registry.exact` is impossible for
    an unsupported operator, and `certify_native` rules it out for a supported
    one.
    """

    def __init__(self, registry, meter=None):
        self.registry = registry
        self.meter = meter or Meter()
        self._mod_cache = {}

    # -- one leaf operator ------------------------------------------------
    def prim(self, op, args):
        n = op.name
        p = dict(op.parameters)
        self.meter.hit(n)
        t = op.output
        if n == "project":
            return args[0][p["index"]]
        if n == "index":
            i = args[1]
            if not 0 <= i < len(args[0]):
                raise IndexError("index outside tuple")
            return args[0][i]
        if n == "tuple":
            return tuple(args)
        if n == "identity":
            return args[0]
        if n == "eq":
            return args[0] == args[1]
        if n in ("lt", "le", "gt", "ge"):
            a, b = args
            return {"lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b}[n]
        if n == "not":
            return not args[0]
        if n in ("and", "or", "xor", "nand", "nor", "xnor"):
            a, b = args
            return {"and": a and b, "or": a or b, "xor": a != b,
                    "nand": not (a and b), "nor": not (a or b), "xnor": a == b}[n]
        if n.startswith("truth_"):
            return bool((int(n[6:]) >> (2 * int(args[0]) + int(args[1]))) & 1)
        if n == "mux":
            return args[1] if args[0] else args[2]
        if n in ("add", "sub", "mul", "min", "max"):
            a, b = args
            v = {"add": a + b, "sub": a - b, "mul": a * b,
                 "min": min(a, b), "max": max(a, b)}[n]
            return _canon_int(t, v)
        if n in ("mod", "idiv"):
            a, b = args
            if b == 0:
                raise ValueError("zero denominator")
            return _canon_int(t, a % b if n == "mod" else a // b)
        if n == "encode":
            x = args[0]
            if t.kind == "bool":
                return bool(x >= p.get("threshold", .5))
            return _canon_int(t, round(float(x)))
        if n == "insert":
            out = args[0] | {args[1]}
            if len(out) > t.capacity:
                raise OverflowError("set capacity exceeded")
            return out
        if n == "remove":
            return args[0] - {args[1]}
        if n == "member":
            return args[1] in args[0]
        if n in ("union", "intersection"):
            out = args[0] | args[1] if n == "union" else args[0] & args[1]
            if len(out) > t.capacity:
                raise OverflowError("set capacity exceeded")
            return out
        if n == "count":
            return _canon_int(t, len(args[0]))
        if n in ("sum", "reduce_min", "reduce_max", "mean"):
            a = sorted(args[0]) if isinstance(args[0], frozenset) else args[0]
            if n != "sum" and not a:
                raise ValueError("empty reduction")
            v = {"sum": lambda: sum(a), "mean": lambda: sum(a) / len(a),
                 "reduce_min": lambda: min(a), "reduce_max": lambda: max(a)}[n]()
            return _canon_int(t, v) if t.kind == "int" and t.encoding.kind == "integer" else v
        if n == "pack":
            raw = 0
            off = 0
            for it, x in zip(op.inputs[0].items, args[0]):
                b = 1 if it.kind == "bool" else it.bits
                raw |= int(x) << off
                off += b
            return _canon_int(t, raw)
        if n == "unpack":
            out = []
            raw = args[0]
            for it in t.items:
                b = 1 if it.kind == "bool" else it.bits
                v = raw & (2 ** b - 1)
                raw >>= b
                out.append(bool(v) if it.kind == "bool" else v)
            return tuple(out)
        raise KeyError("NativeEval cannot execute operator: %s" % n)

    # -- a module call: not a primitive, its leaves are ---------------------
    def call_module(self, name, args):
        m = self.registry.modules[name]
        env = {k: v for (k, _), v in zip(m.inputs, args)}
        for k, v in m.constants:
            env[k] = decode(v.type, v.raw)
        for node in m.nodes:
            c = node.candidates[node.selected]
            env[node.name] = self.apply(c.operator, [env[s] for s in c.sources])
        outs = [env[v] for _, v in m.outputs]
        return outs[0] if len(outs) == 1 else tuple(outs)

    def apply(self, op, args):
        if op.name.startswith("module:"):
            return self.call_module(op.name, args)
        return self.prim(op, args)

    # -- a whole program, eagerly (this is what the algebra does) -----------
    def run(self, program, inputs):
        env = dict(inputs)
        for k, v in program.constants:
            env[k] = decode(v.type, v.raw)
        for node in program.nodes:
            c = node.candidates[node.selected]
            env[node.name] = self.apply(c.operator, [env[s] for s in c.sources])
        return {k: env[v] for k, v in program.outputs}, env


def module_leaf_cost(registry, name, cache=None):
    """Leaf primitive operations inside one module call, statically."""
    cache = cache if cache is not None else {}
    if name in cache:
        return cache[name]
    m = registry.modules[name]
    total = 0
    for node in m.nodes:
        op = node.candidates[node.selected].operator
        total += module_leaf_cost(registry, op.name, cache) if op.name.startswith("module:") else 1
    cache[name] = total
    return total


def certify_native(fx, domain, limit=None):
    """`NativeEval` == `Registry.exact` on every input of the declared domain."""
    p, r = fx["program"], fx["registry"]
    port, t = fx["port"], fx["input_type"]
    ev = NativeEval(r)
    n = 0
    for case in domain:
        native, _ = ev.run(p, {port: case[port]})
        typed, _ = p.run({port: Value.of(t, case[port])}, registry=r)
        typed = {k: v.decoded for k, v in typed.items()}
        if native != typed:
            raise AssertionError("native evaluator diverges at %r: %r vs %r"
                                 % (case, native, typed))
        n += 1
        if limit is not None and n >= limit:
            break
    return n


# ---------------------------------------------------------------------------
# executed bytecodes
# ---------------------------------------------------------------------------
_M = sys.monitoring
_TOOL = 2


def _count(fn):
    n = 0

    def on_instr(code, offset):
        nonlocal n
        n += 1
    _M.use_tool_id(_TOOL, "lazyguard")
    try:
        _M.register_callback(_TOOL, _M.events.INSTRUCTION, on_instr)
        _M.set_events(_TOOL, _M.events.INSTRUCTION)
        fn()
        _M.set_events(_TOOL, 0)
    finally:
        _M.register_callback(_TOOL, _M.events.INSTRUCTION, None)
        _M.free_tool_id(_TOOL)
    return n


_BASE = _count(lambda: None)


def count_opcodes(fn):
    return max(0, _count(fn) - _BASE)


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------
def profile(run_one, domain, meter):
    """Exhaustive worst / expected executed primitives over a uniform domain.

    `run_one(case) -> output`; `meter` is reset around each call.  The domain is
    exhausted, so `expected` is the exact mean under the uniform distribution on
    the declared domain and `worst` is the true maximum.
    """
    total = 0
    worst = 0
    best = None
    count = 0
    hist = {}
    outs = []
    by_total = {}
    for case in domain:
        meter.reset()
        out = run_one(case)
        k = meter.n
        total += k
        worst = max(worst, k)
        best = k if best is None else min(best, k)
        hist[k] = hist.get(k, 0) + 1
        for nm, v in meter.by.items():
            by_total[nm] = by_total.get(nm, 0) + v
        count += 1
        outs.append(out)
    return {"cases": count, "worst_ops": worst, "best_ops": best,
            "expected_ops": total / count, "total_ops": total,
            "histogram": dict(sorted(hist.items())),
            "by_operator_expected": {k: v / count for k, v in sorted(by_total.items())}}, outs


def ratio_table(spec, alg):
    return {"expected_ratio": spec["expected_ops"] / alg["expected_ops"],
            "worst_ratio": spec["worst_ops"] / alg["worst_ops"],
            "best_ratio": spec["best_ops"] / max(1, alg["best_ops"])}
