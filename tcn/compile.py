"""Ahead-of-time compilation of a frozen `Program` to standalone Python source.

PURE ADDITION.  Nothing in `tcn/types.py`, `tcn/graph.py` or `tcn/operators.py`
is touched by this module; the generic interpreter remains the reference and the
oracle.  This file only reads a frozen program and writes Python.

WHY.  `research/inference-cost/RESULTS.md` measured 97.7% of exact execution in
`Type.decode`, `Type.encode` and `validate_raw`, against 0.56% in the operator
semantics and the graph walk.  The cause is structural, not accidental: the
interpreter passes a `Value` across every edge, `Value.__post_init__` validates
the whole carrier recursively on construction, `Value.decoded` re-decodes the
whole carrier on every access, and `Registry.exact` decodes every argument in
full before dispatching.  A 3,072-element raster therefore pays an O(width)
decode and an O(width) validate to execute one O(1) primitive.

WHAT IS ACTUALLY STATIC.  Once a program is frozen, every node has exactly one
candidate, so the operator, its parameters, the type of every operand and the
type of every result are compile-time constants.  Nothing about them can be
learned at run time, so nothing about them needs to be *checked* at run time.
What remains genuinely dynamic is the data-dependent part of the contract --
overflow, division by zero, an out-of-range index, a set over capacity, a
non-finite float, a refinement bound -- and that is preserved exactly, at the
same edge, with the same exception type.

THE REPRESENTATION DECISION.  Generated code carries the *decoded* native Python
value for each SSA name -- `bool`, `int`, `float`, `tuple`, `frozenset` -- and
never a `Value`.  This is chosen, not assumed, and it rests on one fact:

    for a valid carrier, `encode(t, decode(t, raw)) == raw`

so `decode . encode` is *idempotent* on canonical values.  Every SSA name in the
generated code is canonical by construction, which means an operator that only
rearranges already-canonical parts -- `tuple`, `project`, `index`, `mux`,
`identity`, `member`, the set algebra, `map`, `filter`, a module call -- needs no
re-encoding at all; only the O(1) checks that can actually fail are emitted.
Only an operator that computes a *new* scalar has to re-apply the encoding, and
for a statically known scalar type that is a couple of comparisons inline.
Large structured values are therefore passed as **immutable references** to
Python tuples and frozensets.  No copy, no view object, no index carrier: a
Python tuple already gives O(1) element access and free structural sharing
because it is immutable, which is the same property the type system relies on.

SCOPE OF EQUIVALENCE, STATED PLAINLY.  Errors that `Program.validate` proves
unreachable on a frozen program -- an operand of the wrong carrier kind, a tuple
of the wrong arity, an index constant outside a statically known tuple -- are
not re-checked on internal edges.  Values entering the program at an input port
are validated in full, exactly as `Value.of` does.  Every data-dependent error
is preserved: `OverflowError` on integer range, on float packing and on set
capacity; `ValueError` on a fractional value, a zero denominator, a shift
outside the width, an empty reduction and a refinement bound; `IndexError` on a
dynamic index; `TypeError` on a non-finite intermediate.

Pipeline: validate once -> prune -> intern types -> lower to straight-line SSA
-> constant-fold and eliminate dead code where exact -> emit.
"""
from __future__ import annotations

import json
import math

from .graph import Program
from .operators import BINARY, COMPARE, CONVERSIONS, LOGIC, Registry, UNARY
from .types import Type, Value, decode

__all__ = ["compile_program", "CompileResult"]

_MATH_BINARY = {"pow": "_pow", "atan2": "_atan2"}
_MATH_UNARY = {"exp": "_exp", "log": "_log", "sin": "_sin", "cos": "_cos", "sqrt": "_sqrt"}
_INT_CLOSED = {"add", "sub", "mul", "min", "max", "mod", "idiv", "shl", "shr", "neg", "abs"}
_UNKNOWN = object()
_FLOAT_CLOSED = {"add", "sub", "mul", "div", "pow", "min", "max", "atan2",
                 "neg", "abs", "exp", "log", "sin", "cos", "sqrt"}


class CompileResult:
    """Generated source plus the audit trail from each line back to a node."""

    def __init__(self, source, provenance, types, stats):
        self.source = source
        self.provenance = provenance
        self.types = types
        self.stats = stats

    def module(self, name="tcn_compiled"):
        """Import the generated source as a module object (for testing)."""
        import types as _t
        mod = _t.ModuleType(name)
        mod.__dict__["__file__"] = f"<{name}>"
        exec(compile(self.source, f"<{name}>", "exec"), mod.__dict__)
        return mod


def _tkey(t):
    return json.dumps(t.to_dict(), sort_keys=True, separators=(",", ":"))


def _lit(x):
    """A Python literal for a decoded value.  Deterministic for sets."""
    if isinstance(x, bool):
        return repr(x)
    if isinstance(x, frozenset):
        if not x:
            return "frozenset()"
        return "frozenset([%s])" % ",".join(sorted(_lit(v) for v in x))
    if isinstance(x, tuple):
        inner = ",".join(_lit(v) for v in x)
        return "(%s,)" % inner if len(x) == 1 else "(%s)" % inner
    return repr(x)


class _Compiler:
    def __init__(self, program, registry, entry="run"):
        self.registry = registry
        self.entry = entry
        self.program = program
        self.types = {}            # canonical type key -> index
        self.type_dicts = []       # index -> canonical dict
        self.canon = {}            # type index -> helper name (scalar canonicaliser)
        self.bound = {}            # type index -> helper name (boundary encoder)
        self.helpers = []          # helper function sources, in emission order
        self.consts = {}           # literal source -> constant name
        self.known = {}            # constant name -> its value, for guard folding
        self.const_lines = []
        self.modules = {}          # module name -> function name
        self.module_lines = []
        self.prov = {}             # variable name -> provenance record
        self.n = 0
        self.stats = {"nodes_emitted": 0, "nodes_folded": 0, "nodes_pruned_dead": 0,
                      "scalar_canonicalisations": 0, "structural_canonicalisations_elided": 0,
                      "boundary_encoders": 0, "modules_emitted": 0, "widest_static_type": 0}
        self.needs = set()

    # ---- type table -------------------------------------------------------
    def tid(self, t):
        """Intern a type and return its index in the canonical type table.

        Nested types are stored *by index*, not inline.  `Type.to_dict` writes a
        3,072-field tuple as 3,072 separate JSON objects at every mention, which
        `research/inference-cost/RESULTS.md` sec 5.1 measured as 99.68% of the
        exported artifact; one shared table keyed by the type's own canonical
        JSON collapses that, and it is also what makes a structural type compare
        an identity test rather than a deep walk.
        """
        key = _tkey(t)
        i = self.types.get(key)
        if i is not None:
            return i
        items = [self.tid(x) for x in t.items]
        key = _tkey(t)
        i = self.types.get(key)
        if i is not None:
            return i
        d = {"kind": t.kind}
        if t.kind == "int":
            d["bits"] = t.bits
            d["encoding"] = {"kind": t.encoding.kind, "signed": t.encoding.signed,
                             "scale": t.encoding.scale, "overflow": t.encoding.overflow}
        if items:
            d["items"] = items
        if t.kind == "set":
            d["capacity"] = t.capacity
        for k in ("role", "unit", "frame", "bounds"):
            if getattr(t, k):
                d[k] = getattr(t, k)
        i = len(self.type_dicts)
        self.types[key] = i
        self.type_dicts.append(d)
        self.stats["widest_static_type"] = max(self.stats["widest_static_type"], t.width)
        return i

    def const(self, value):
        src = _lit(value)
        name = self.consts.get(src)
        if name is None:
            name = "_k%d" % len(self.consts)
            self.consts[src] = name
            self.const_lines.append("%s = %s" % (name, src))
            self.known[name] = value
        return name

    def var(self):
        self.n += 1
        return "v%d" % self.n

    # ---- scalar canonicalisation ------------------------------------------
    def _canon_fn(self, t):
        """`decode(t, encode(t, x))` for a scalar type, partially evaluated."""
        i = self.tid(t)
        fn = self.canon.get(i)
        if fn is not None:
            return fn
        fn = "_c%d" % i
        self.canon[i] = fn
        self.helpers.append("def %s(x):\n%s\n" % (fn, "\n".join(self._canon_body(t))))
        return fn

    def _canon_body(self, t, var="x"):
        L = []
        if t.kind == "bool":
            L.append("    if type(x) is not bool: raise TypeError('Boolean encoding requires bool, not numeric coercion')")
            L.append("    return x")
            return L
        e = t.encoding
        self.needs.add("isfinite")
        L.append("    if not _isfinite(x): _nonfinite()")
        if t.bounds is not None:
            L.append("    if not %r <= x <= %r: raise ValueError('value outside semantic bounds')"
                     % (t.bounds[0], t.bounds[1]))
        if e.kind == "float":
            if t.bits == 32:
                self.needs.add("f32")
                L.append("    d = _F32U(_F32P(x))[0]")
            else:
                L.append("    d = float(x)")
            L.append("    if not _isfinite(d): raise OverflowError('nonfinite float encoding')")
            if t.bounds is not None:
                L.append("    if not %r <= d <= %r: raise ValueError('encoded value violates domain')"
                         % (t.bounds[0], t.bounds[1]))
            L.append("    return d")
            return L
        M = 2 ** t.bits
        lo = -(2 ** (t.bits - 1)) if e.signed else 0
        hi = 2 ** (t.bits - int(e.signed)) - 1
        if e.kind == "fixed":
            L.append("    n = round(x * %r)" % e.scale)
        else:
            L.append("    n = int(x)")
            L.append("    if n != x: raise ValueError('fractional value requires explicit quantization')")
        if e.overflow == "error":
            L.append("    if not %d <= n <= %d: raise OverflowError('%%s outside [%d, %d]' %% n)" % (lo, hi, lo, hi))
        elif e.overflow == "saturate":
            L.append("    if n < %d: n = %d" % (lo, lo))
            L.append("    elif n > %d: n = %d" % (hi, hi))
        if e.overflow != "error":
            L.append("    n = n %% %d" % M)
            if e.signed:
                L.append("    if n >= %d: n = n - %d" % (M // 2, M))
        if e.kind == "fixed":
            L.append("    d = n / %r" % e.scale)
        else:
            L.append("    d = n")
        if t.bounds is not None:
            L.append("    if not %r <= d <= %r: raise ValueError('encoded value violates domain')"
                     % (t.bounds[0], t.bounds[1]))
        L.append("    return d")
        return L

    # ---- boundary encoders ------------------------------------------------
    def _boundary_fn(self, t):
        """Full `Value.of` semantics: encode with every guard, then decode."""
        i = self.tid(t)
        fn = self.bound.get(i)
        if fn is not None:
            return fn
        fn = "_b%d" % i
        self.bound[i] = fn
        self.bound[i] = fn
        self.stats["boundary_encoders"] += 1
        if t.kind == "tuple":
            inner = [self._boundary_fn(s) for s in t.items]
            body = ["    if len(x) != %d: raise ValueError('tuple arity mismatch')" % len(t.items)]
            if len(set(inner)) == 1 and len(inner) > 4:
                body.append("    return tuple(map(%s, x))" % inner[0])
            else:
                body.append("    return (%s)" % "".join("%s(x[%d])," % (f, k) for k, f in enumerate(inner)))
        elif t.kind == "set":
            e = self._boundary_fn(t.items[0])
            body = ["    out = frozenset(map(%s, x))" % e,
                    "    if len(out) > %d: raise OverflowError('set capacity exceeded')" % t.capacity,
                    "    return out"]
        elif t.kind == "bool":
            body = ["    if type(x) is not bool: raise TypeError('Boolean encoding requires bool, not numeric coercion')",
                    "    return x"]
        else:
            body = ["    if not isinstance(x, (int, float)) or isinstance(x, bool): _nonfinite()",
                    "    return %s(x)" % self._canon_fn(t)]
        self.helpers.append("def %s(x):\n%s\n" % (fn, "\n".join(body)))
        return fn

    # ---- raw <-> decoded for pack / unpack --------------------------------
    def _to_raw(self, t, expr):
        if t.kind == "bool":
            return "int(%s)" % expr
        e = t.encoding
        M = 2 ** t.bits
        if e.kind == "float":
            self.needs.add("f32" if t.bits == 32 else "f64")
            return ("_U32U(_F32P(%s))[0]" if t.bits == 32 else "_U64U(_F64P(%s))[0]") % expr
        if e.kind == "fixed":
            return "(round(%s * %r) %% %d)" % (expr, e.scale, M)
        return "(%s %% %d)" % (expr, M) if e.signed else "(%s)" % expr

    def _from_raw(self, t, expr):
        if t.kind == "bool":
            return "bool(%s)" % expr
        e = t.encoding
        M = 2 ** t.bits
        if e.kind == "float":
            self.needs.add("f32" if t.bits == 32 else "f64")
            return ("_F32U(_U32P(%s))[0]" if t.bits == 32 else "_F64U(_U64P(%s))[0]") % expr
        body = "(%s)" % expr
        if e.signed:
            body = "(%s - %d if %s >= %d else %s)" % (expr, M, expr, M // 2, expr)
        if e.kind == "fixed":
            body = "(%s / %r)" % (body, e.scale)
        return body

    # ---- canonicalisation strategy ---------------------------------------
    def _fast_kind(self, op):
        """Can the encode/decode round trip be inlined as one range test?"""
        t = op.output
        if t.kind != "int" or t.bounds is not None:
            return None
        e = t.encoding
        ins = [i for i in op.inputs if i.kind == "int"]
        if e.kind == "integer" and e.overflow == "error":
            if op.name in _INT_CLOSED and all(i.encoding.kind == "integer" for i in ins):
                return "int"
            if op.name == "count":
                return "int"
            if op.name in CONVERSIONS and op.name != "pack" and op.name != "unpack":
                # `round(float(x))` is already an exact int; the interpreter's
                # `float(x)` raises before `Value.of` sees the value, and the
                # generated expression keeps that order.
                return "int"
            if (op.name in {"sum", "reduce_min", "reduce_max"} and op.inputs[0].kind == "tuple"
                    and all(i.encoding.kind == "integer" for i in op.inputs[0].items)):
                return "int"
        if e.kind == "float" and t.bits == 64 and op.name in _FLOAT_CLOSED:
            if all(i.encoding.kind == "float" and i.bits == 64 for i in ins):
                return "float64"
        return None

    def _emit_scalar(self, dst, expr, op):
        """Assign `dst` the canonical form of `expr` under `op.output`."""
        t = op.output
        if t.kind == "bool":
            return ["%s = %s" % (dst, expr)]
        kind = self._fast_kind(op)
        if kind == "int":
            M = 2 ** t.bits
            lo = -(M // 2) if t.encoding.signed else 0
            hi = M // (2 if t.encoding.signed else 1) - 1
            self.needs.add("ovf")
            self.stats["scalar_canonicalisations"] += 1
            return ["%s = %s" % (dst, expr),
                    "if not %d <= %s <= %d: _ovf(%s, %d, %d)" % (lo, dst, hi, dst, lo, hi)]
        if kind == "float64":
            self.needs.add("isfinite")
            self.stats["scalar_canonicalisations"] += 1
            return ["%s = %s" % (dst, expr),
                    "if not _isfinite(%s): _nonfinite()" % dst]
        self.stats["scalar_canonicalisations"] += 1
        return ["%s = %s(%s)" % (dst, self._canon_fn(t), expr)]

    # ---- operator lowering ------------------------------------------------
    def _lower(self, dst, op, s, scope):
        """Straight-line Python for one frozen operator application."""
        n = op.name
        p = dict(op.parameters)
        out = op.output
        L = []

        if n.startswith("module:"):
            fn = self.emit_module(n)
            self.stats["structural_canonicalisations_elided"] += 1
            return ["%s = %s(%s)" % (dst, fn, ", ".join(s))]
        if n == "identity" or n == "interpret":
            # `interpret` moves `role` only; `Registry.resolve` requires the
            # carrier, encoding, unit, frame and bounds to be preserved, so the
            # decoded value is bit-for-bit the same object.
            self.stats["structural_canonicalisations_elided"] += 1
            return ["%s = %s" % (dst, s[0])]
        if n == "not":
            return ["%s = not %s" % (dst, s[0])]
        if n in LOGIC:
            e = {"and": "%s and %s", "or": "%s or %s", "xor": "%s != %s",
                 "nand": "not (%s and %s)", "nor": "not (%s or %s)", "xnor": "%s == %s"}[n]
            return ["%s = %s" % (dst, e % (s[0], s[1]))]
        if n.startswith("truth_"):
            k = int(n[6:])
            table = tuple(bool((k >> j) & 1) for j in range(4))
            return ["%s = %s[2 * %s + %s]" % (dst, self.const(table), s[0], s[1])]
        if n == "mux":
            self.stats["structural_canonicalisations_elided"] += 1
            return ["%s = %s if %s else %s" % (dst, s[1], s[0], s[2])]
        if n in COMPARE:
            e = {"eq": "==", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}[n]
            return ["%s = %s %s %s" % (dst, s[0], e, s[1])]
        if n in BINARY:
            k = self.known.get(s[1], _UNKNOWN)
            if n in {"div", "mod", "idiv"}:
                if k is _UNKNOWN:
                    L.append("if %s == 0: raise ValueError('zero denominator')" % s[1])
                elif k == 0:
                    return ["raise ValueError('zero denominator')"]
            if n in {"shl", "shr"}:
                if k is _UNKNOWN:
                    L.append("if not 0 <= %s < %d: raise ValueError('shift outside bit width')"
                             % (s[1], op.inputs[0].bits))
                elif not 0 <= k < op.inputs[0].bits:
                    return ["raise ValueError('shift outside bit width')"]
            sym = {"add": "+", "sub": "-", "mul": "*", "div": "/", "mod": "%",
                   "idiv": "//", "shl": "<<", "shr": ">>"}.get(n)
            if sym:
                expr = "%s %s %s" % (s[0], sym, s[1])
            elif n in _MATH_BINARY:
                self.needs.add(n)
                expr = "%s(%s, %s)" % (_MATH_BINARY[n], s[0], s[1])
            else:
                expr = "%s(%s, %s)" % (n, s[0], s[1])          # min / max
            return L + self._emit_scalar(dst, expr, op)
        if n in UNARY:
            if n == "neg":
                expr = "-%s" % s[0]
            elif n == "abs":
                expr = "abs(%s)" % s[0]
            else:
                self.needs.add(n)
                expr = "%s(%s)" % (_MATH_UNARY[n], s[0])
            return self._emit_scalar(dst, expr, op)
        if n in {"sum", "mean", "reduce_min", "reduce_max", "count"}:
            src = s[0]
            if op.inputs[0].kind == "set":
                tmp = self.var()
                L.append("%s = sorted(%s)" % (tmp, src))
                src = tmp
            if n in {"mean", "reduce_min", "reduce_max"}:
                L.append("if not %s: raise ValueError('empty reduction')" % src)
            expr = {"sum": "sum(%s)", "mean": "sum(%s) / len(%s)",
                    "reduce_min": "min(%s)", "reduce_max": "max(%s)", "count": "len(%s)"}[n]
            expr = expr % ((src, src) if n == "mean" else src)
            return L + self._emit_scalar(dst, expr, op)
        if n == "tuple":
            self.stats["structural_canonicalisations_elided"] += 1
            return ["%s = (%s)" % (dst, "".join(x + "," for x in s))]
        if n == "project":
            self.stats["structural_canonicalisations_elided"] += 1
            return ["%s = %s[%d]" % (dst, s[0], p["index"])]
        if n == "index":
            self.stats["structural_canonicalisations_elided"] += 1
            k = self.known.get(s[1], _UNKNOWN)
            if k is _UNKNOWN:
                L.append("if not 0 <= %s < %d: raise IndexError('index outside tuple')"
                         % (s[1], len(op.inputs[0].items)))
            elif not 0 <= k < len(op.inputs[0].items):
                return ["raise IndexError('index outside tuple')"]
            return L + ["%s = %s[%s]" % (dst, s[0], s[1])]
        if n == "member":
            return ["%s = %s in %s" % (dst, s[1], s[0])]
        if n in {"insert", "union"}:
            self.stats["structural_canonicalisations_elided"] += 1
            expr = "%s | {%s}" % (s[0], s[1]) if n == "insert" else "%s | %s" % (s[0], s[1])
            return ["%s = %s" % (dst, expr),
                    "if len(%s) > %d: raise OverflowError('set capacity exceeded')" % (dst, out.capacity)]
        if n == "remove":
            self.stats["structural_canonicalisations_elided"] += 1
            return ["%s = %s - {%s}" % (dst, s[0], s[1])]
        if n == "intersection":
            self.stats["structural_canonicalisations_elided"] += 1
            return ["%s = %s & %s" % (dst, s[0], s[1])]
        if n in {"pair", "join"}:
            self.stats["structural_canonicalisations_elided"] += 1
            a, b = self.var(), self.var()
            cond = "" if n == "pair" else " if %s[%d] == %s[%d]" % (a, p["left"], b, p["right"])
            return ["%s = frozenset((%s, %s) for %s in %s for %s in %s%s)"
                    % (dst, a, b, a, s[0], b, s[1], cond),
                    "if len(%s) > %d: raise OverflowError('set capacity exceeded')" % (dst, out.capacity)]
        if n in {"map", "filter"}:
            fn = self.emit_module(p["module"])
            x = self.var()
            self.stats["structural_canonicalisations_elided"] += 1
            if n == "map":
                return ["%s = frozenset(%s(%s) for %s in %s)" % (dst, fn, x, x, s[0])]
            return ["%s = frozenset(%s for %s in %s if %s(%s))" % (dst, x, x, s[0], fn, x)]
        if n == "delay":
            self.stats["structural_canonicalisations_elided"] += 1
            return ["%s = (%s, %s)" % (dst, s[1], s[0])]
        if n in {"difference", "accumulation"}:
            scalar = _ScalarOp(op.output.items[0], op.inputs[0], "sub" if n == "difference" else "add")
            tmp = self.var()
            L += self._emit_scalar(tmp, "%s - %s" % (s[0], s[1]) if n == "difference"
                                   else "%s + %s" % (s[1], s[0]), scalar)
            return L + ["%s = (%s, %s)" % (dst, tmp, s[0] if n == "difference" else tmp)]
        if n in {"fft", "ifft"}:
            self.needs.add("cmath")
            size = len(op.inputs[0].items)
            # both directions return a tuple of (real, imag) pairs
            comp = op.output.items[0].items[0]
            src = ("[complex(_z) for _z in %s]" % s[0] if n == "fft"
                   else "[complex(_z[0], _z[1]) for _z in %s]" % s[0])
            sign = -1 if n == "fft" else 1
            scale = "" if n == "fft" else " / %d" % size
            a = self.var()
            L.append("%s = %s" % (a, src))
            z = self.var()
            L.append("%s = [sum(_v * _cexp(%d * 2j * _pi * _k * _j / %d) for _j, _v in enumerate(%s))%s "
                     "for _k in range(%d)]" % (z, sign, size, a, scale, size))
            fn = self._canon_fn(comp)
            self.stats["scalar_canonicalisations"] += 2 * size
            L.append("%s = tuple((%s(_v.real), %s(_v.imag)) for _v in %s)" % (dst, fn, fn, z))
            return L
        if n == "fourier_basis":
            self.needs.add("cos")
            self.needs.add("sin")
            fn = self._canon_fn(op.output.items[0])
            self.stats["scalar_canonicalisations"] += 2
            return ["%s = (%s(_cos(%s)), %s(_sin(%s)))" % (dst, fn, s[0], fn, s[0])]
        if n == "pack":
            items = op.inputs[0].items
            parts, off = [], 0
            for k, t in enumerate(items):
                parts.append("(%s << %d)" % (self._to_raw(t, "%s[%d]" % (s[0], k)), off))
                off += 1 if t.kind == "bool" else t.bits
            raw = self.var()
            L.append("%s = %s" % (raw, " | ".join(parts) if parts else "0"))
            L.append("%s = %s" % (dst, self._from_raw(out, raw)))
            if out.bounds is not None:
                L.append("if not %r <= %s <= %r: raise ValueError('encoded value violates domain')"
                         % (out.bounds[0], dst, out.bounds[1]))
            return L
        if n == "unpack":
            raw = self.var()
            L.append("%s = %s" % (raw, self._to_raw(op.inputs[0], s[0])))
            parts, off = [], 0
            for t in out.items:
                b = 1 if t.kind == "bool" else t.bits
                parts.append(self._from_raw(t, "((%s >> %d) & %d)" % (raw, off, 2 ** b - 1)))
                off += b
            L.append("%s = (%s)" % (dst, "".join(x + "," for x in parts)))
            for k, t in enumerate(out.items):
                if t.bounds is not None:
                    L.append("if not %r <= %s[%d] <= %r: raise ValueError('encoded value violates domain')"
                             % (t.bounds[0], dst, k, t.bounds[1]))
            return L
        if n in CONVERSIONS:
            if out.kind == "bool":
                expr = "bool(%s >= %r)" % (s[0], p.get("threshold", .5))
            elif out.encoding.kind == "integer":
                expr = "round(float(%s))" % s[0]
            else:
                expr = "float(%s)" % s[0]
            return self._emit_scalar(dst, expr, op)
        raise KeyError("cannot compile operator: %s" % n)

    # ---- program bodies ---------------------------------------------------
    def emit_module(self, name):
        fn = self.modules.get(name)
        if fn is not None:
            return fn
        module = self.registry.modules[name]
        fn = "_m%d" % len(self.modules)
        self.modules[name] = fn
        self.stats.setdefault("module_function", {})[name] = fn
        self.stats["modules_emitted"] += 1
        ports = [self.var() for _ in module.inputs]
        env = {k: v for (k, _), v in zip(module.inputs, ports)}
        for k, v in zip(module.inputs, ports):
            self.prov[v] = {"scope": name, "node": k[0], "role": "input", "type": self.tid(k[1])}
        body = self.emit_body(module, env, name)
        outs = [env[v] for _, v in module.outputs]
        ret = outs[0] if len(outs) == 1 else "(%s)" % "".join(x + "," for x in outs)
        src = ["def %s(%s):" % (fn, ", ".join(ports)),
               "    # %s -- %d nodes" % (name, len(module.nodes))]
        src += ["    " + line for line in body]
        src.append("    return %s" % ret)
        src.append("")
        self.module_lines.append("\n".join(src))
        return fn

    def emit_body(self, program, env, scope):
        """Lower a program's nodes into straight-line statements.

        Constant folding and dead-code elimination run here, over the frozen
        selection only, so nothing that could change semantics is removed.
        """
        program = program.pruned()
        folded = {}
        const_vals = dict(program.constants)
        for n in program.nodes:
            c = n.candidates[n.selected]
            if all(sc in const_vals for sc in c.sources):
                try:
                    const_vals[n.name] = self.registry.exact(c.operator, [const_vals[k] for k in c.sources])
                    folded[n.name] = const_vals[n.name]
                    continue
                except Exception:
                    pass                      # a node that raises is left in place
        # dead-code elimination over the folded graph
        by = {n.name: n for n in program.nodes}
        keep, stack = set(), [v for _, v in program.outputs] + [u for _, _, u in program.state]
        while stack:
            k = stack.pop()
            if k in by and k not in keep and k not in folded:
                keep.add(k)
                stack += list(by[k].candidates[by[k].selected].sources)
        self.stats["nodes_folded"] += len(folded)
        self.stats["nodes_pruned_dead"] += len(program.nodes) - len(keep) - len(folded)

        for k, v in program.constants:
            if k not in env:
                env[k] = self.const(decode(v.type, v.raw))
        for k, v in folded.items():
            env[k] = self.const(decode(v.type, v.raw))
            self.prov[env[k]] = {"scope": scope, "node": k, "role": "constant-folded",
                                 "type": self.tid(v.type)}
        lines = []
        for n in program.nodes:
            if n.name not in keep:
                continue
            c = n.candidates[n.selected]
            dst = self.var()
            srcs = [env[k] for k in c.sources]
            stmts = self._lower(dst, c.operator, srcs, scope)
            tag = "%s/%s := %s(%s)" % (scope, n.name, c.operator.name, ", ".join(c.sources))
            lines.append("# " + tag)
            lines += stmts
            env[n.name] = dst
            self.stats["nodes_emitted"] += 1
            self.stats.setdefault("emitted_per_scope", {})
            self.stats["emitted_per_scope"][scope] = self.stats["emitted_per_scope"].get(scope, 0) + 1
            self.prov[dst] = {"scope": scope, "node": n.name, "operator": c.operator.name,
                              "parameters": dict(c.operator.parameters),
                              "sources": list(c.sources), "type": self.tid(n.output)}
        return lines

    # ---- top level --------------------------------------------------------
    def build(self):
        p = self.program.pruned()
        p.validate(self.registry)
        if not p.is_frozen:
            raise ValueError("compilation requires a fully crystallized program")

        env, arg_lines = {}, []
        for k, t in p.inputs:
            v = self.var()
            env[k] = v
            self.prov[v] = {"scope": "run", "node": k, "role": "input", "type": self.tid(t)}
            arg_lines.append("    %s = _IN[%r](inputs[%r]) if validate else inputs[%r]"
                             % (v, k, k, k))
        state_names = [k for k, _, _ in p.state]
        for k, v0, _ in p.state:
            var = self.var()
            env[k] = var
            self.prov[var] = {"scope": "run", "node": k, "role": "state", "type": self.tid(v0.type)}
            arg_lines.append("    %s = _S0[%r] if state is None else (_ST[%r](state[%r]) if validate else state[%r])"
                             % (var, k, k, k, k))
        body = self.emit_body(p, env, "run")

        # boundary encoders, built after the body so only reachable types appear
        # and before the header so their helper requirements are known
        in_map = {k: self._boundary_fn(t) for k, t in p.inputs}
        st_map = {k: self._boundary_fn(v.type) for k, v, _ in p.state}
        s0_map = {k: self.const(decode(v.type, v.raw)) for k, v, _ in p.state}

        header = ['"""Generated by tcn.compile -- do not edit.',
                  "",
                  "Frozen program digest: %s" % p.digest,
                  "Standard library only: no torch, no numpy, no tcn.",
                  "",
                  "Every statement carries the typed node and operator it came from, both as a",
                  "source comment and in PROVENANCE.  TYPES is the canonical interned type table;",
                  "PROVENANCE values reference it by index.",
                  '"""',
                  "import math"]
        if "cmath" in self.needs:
            header.append("import cmath")
            header.append("_cexp = cmath.exp")
        if self.needs & {"f32", "f64"}:
            header.append("import struct")
        header.append("_isfinite = math.isfinite")
        header.append("_pi = math.pi")
        for k, alias in list(_MATH_UNARY.items()) + list(_MATH_BINARY.items()):
            if k in self.needs:
                header.append("%s = math.%s" % (alias, k))
        if "f32" in self.needs:
            header += ["_F32P = struct.Struct('<f').pack", "_F32U = struct.Struct('<f').unpack",
                       "_U32P = struct.Struct('<I').pack", "_U32U = struct.Struct('<I').unpack"]
        if "f64" in self.needs:
            header += ["_F64P = struct.Struct('<d').pack", "_F64U = struct.Struct('<d').unpack",
                       "_U64P = struct.Struct('<Q').pack", "_U64U = struct.Struct('<Q').unpack"]
        header += ["", "def _nonfinite(): raise TypeError('finite numeric value required')", ""]
        if "ovf" in self.needs:
            header += ["def _ovf(x, lo, hi):",
                       "    if not _isfinite(x): _nonfinite()",
                       "    raise OverflowError('%s outside [%s, %s]' % (x, lo, hi))", ""]

        run = ["def run(inputs, state=None, validate=True):",
               '    """Batch-one exact execution.  Native decoded values in and out."""']
        run += arg_lines
        run += ["    " + line for line in body]
        run.append("    return ({%s}, {%s})" % (
            ", ".join("%r: %s" % (k, env[v]) for k, v in p.outputs),
            ", ".join("%r: %s" % (k, env[u]) for k, _, u in p.state)))

        parts = ["\n".join(header), "\n".join(self.helpers), "\n".join(self.const_lines), ""]
        parts.append("_IN = {%s}" % ", ".join("%r: %s" % (k, v) for k, v in in_map.items()))
        parts.append("_ST = {%s}" % ", ".join("%r: %s" % (k, v) for k, v in st_map.items()))
        parts.append("_S0 = {%s}" % ", ".join("%r: %s" % (k, v) for k, v in s0_map.items()))
        parts.append("DIGEST = %r" % p.digest)
        parts.append("INPUTS = %r" % [[k, self.tid(t)] for k, t in p.inputs])
        parts.append("OUTPUTS = %r" % [list(o) for o in p.outputs])
        parts.append("STATE = %r" % state_names)
        parts.append("TYPES = %r" % (self.type_dicts,))
        parts.append("PROVENANCE = %r" % (self.prov,))
        parts.append("")
        parts.append("\n".join(self.module_lines))
        parts.append("\n".join(run))
        parts.append("")
        return "\n".join(parts)


class _ScalarOp:
    """A synthetic operator record, so `difference`/`accumulation` can reuse the
    scalar canonicalisation path for the arithmetic component they compute."""

    def __init__(self, output, operand, name):
        self.output = output
        self.inputs = (operand, operand)
        self.name = name
        self.parameters = ()


def compile_program(program, registry=None, entry="run"):
    """Compile a frozen `Program` to standalone Python source.

    Returns a `CompileResult` with `.source`, `.provenance`, `.types`, `.stats`.
    The interpreter is untouched and remains the oracle for equivalence.
    """
    registry = registry or Registry()
    c = _Compiler(program, registry, entry)
    source = c.build()
    c.stats["type_table_entries"] = len(c.type_dicts)
    c.stats["source_bytes"] = len(source.encode())
    c.stats["source_lines"] = source.count("\n") + 1
    return CompileResult(source, c.prov, c.type_dicts, c.stats)
