"""A tiny algorithm IR: `guard`, `find`, `scan`, `let` -- and nothing else.

DESIGN sec 3 asks for exactly what the measured gap requires and no more.  This
is that IR.  It is a *second* language: the training algebra in `tcn/` is
untouched and remains the specification and the oracle.

    Lit / Ref / Var / Prim / Call        -- the specification's own vocabulary
    Let   name = e in body               -- sharing
    Guard c then a else b                -- LAZY: only the taken branch runs
    Find  v in [lo,hi) where p -> t | d  -- first index satisfying p, else d
    Scan  v in [lo,hi) from a0 by f      -- bounded accumulation, optional `while`

`Find` and `Scan` carry an explicit finite bound, so termination is structural.

COST.  One meter hit per leaf primitive, exactly as `cost.NativeEval` charges
the specification, so the two are commensurable.  A `Guard` costs 1, the same as
the `mux` it can replace.  One `Find` or `Scan` iteration test costs 1, the same
as the `mux` in an unrolled chain.  These choices are what make the worst case
come out *equal* rather than better, which is the honest outcome.

EVALUATION.  Bindings are call-by-need and memoised.  That alone changes
nothing: every argument of a `Prim` is strict, so an unmodified specification
lowered into this IR forces every node.  `control_lazy_evaluator` in
`stage1.py` measures exactly that, and it is the control that keeps the gain
attributable to the `Guard` construct rather than to the evaluation strategy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cost as _cost


# ---------------------------------------------------------------------------
# syntax
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Lit:
    value: object
    tag: str = ""

    def __repr__(self):
        return "Lit(%r)" % (self.value,)


@dataclass(frozen=True)
class Ref:
    name: str

    def __repr__(self):
        return "Ref(%s)" % self.name


@dataclass(frozen=True)
class Var:
    name: str

    def __repr__(self):
        return "Var(%s)" % self.name


@dataclass(frozen=True)
class Prim:
    op: object                       # a tcn Operator
    args: tuple

    def __repr__(self):
        return "%s(%s)" % (self.op.name, ", ".join(map(repr, self.args)))


@dataclass(frozen=True)
class Call:
    module: str
    args: tuple

    def __repr__(self):
        return "%s(%s)" % (self.module[:13], ", ".join(map(repr, self.args)))


@dataclass(frozen=True)
class Guard:
    cond: object
    then: object
    other: object

    def __repr__(self):
        return "guard(%r, %r, %r)" % (self.cond, self.then, self.other)


@dataclass(frozen=True)
class Let:
    name: str
    value: object
    body: object

    def __repr__(self):
        return "let %s = %r in %r" % (self.name, self.value, self.body)


@dataclass(frozen=True)
class Find:
    var: str
    lo: int
    hi: int
    pred: object
    then: object
    default: object

    def __repr__(self):
        return "find %s in [%d,%d) where %r -> %r else %r" % (
            self.var, self.lo, self.hi, self.pred, self.then, self.default)


@dataclass(frozen=True)
class Scan:
    var: str
    acc: str
    lo: int
    hi: int
    init: object
    step: object
    while_: object = None

    def __repr__(self):
        w = "" if self.while_ is None else " while %r" % (self.while_,)
        return "scan %s in [%d,%d) from %r by (%s -> %r)%s" % (
            self.var, self.lo, self.hi, self.init, self.acc, self.step, w)


@dataclass
class IRProgram:
    """Lazy bindings plus an output expression."""
    bindings: list                    # [(name, expr)] in dependency order
    output: object
    inputs: tuple = ()

    def rebind(self, name, expr):
        b = [(k, expr if k == name else v) for k, v in self.bindings]
        return IRProgram(b, self.output, self.inputs)

    def size(self):
        return sum(_size(e) for _, e in self.bindings) + _size(self.output)

    def constructs(self):
        s = set()
        for _, e in self.bindings:
            _constructs(e, s)
        _constructs(self.output, s)
        return s


def _size(e):
    if isinstance(e, (Lit, Ref, Var)):
        return 1
    if isinstance(e, (Prim, Call)):
        return 1 + sum(_size(a) for a in e.args)
    if isinstance(e, Guard):
        return 1 + _size(e.cond) + _size(e.then) + _size(e.other)
    if isinstance(e, Let):
        return 1 + _size(e.value) + _size(e.body)
    if isinstance(e, Find):
        return 1 + _size(e.pred) + _size(e.then) + _size(e.default)
    if isinstance(e, Scan):
        return 1 + _size(e.init) + _size(e.step) + (0 if e.while_ is None else _size(e.while_))
    raise TypeError(type(e))


def _constructs(e, s):
    s.add(type(e).__name__)
    for a in _children(e):
        _constructs(a, s)


def _children(e):
    if isinstance(e, (Prim, Call)):
        return list(e.args)
    if isinstance(e, Guard):
        return [e.cond, e.then, e.other]
    if isinstance(e, Let):
        return [e.value, e.body]
    if isinstance(e, Find):
        return [e.pred, e.then, e.default]
    if isinstance(e, Scan):
        return [e.init, e.step] + ([] if e.while_ is None else [e.while_])
    return []


def necessary(e, table, seen=None):
    """Bindings certainly forced when `e` is evaluated.  Conservative.

    A `Guard` contributes its condition plus the *intersection* of its branches;
    a `Find` or `Scan` body may not run at all, so only the initial value counts.
    """
    seen = set() if seen is None else seen
    t = type(e)
    if t is Ref:
        if e.name in seen or e.name not in table:
            return {e.name} if e.name in table else set()
        return {e.name} | necessary(table[e.name], table, seen | {e.name})
    if t in (Prim, Call):
        out = set()
        for a in e.args:
            out |= necessary(a, table, seen)
        return out
    if t is Guard:
        return (necessary(e.cond, table, seen)
                | (necessary(e.then, table, seen) & necessary(e.other, table, seen)))
    if t is Let:
        return necessary(e.value, table, seen) | necessary(e.body, table, seen)
    if t is Find:
        return set()
    if t is Scan:
        return necessary(e.init, table, seen)
    return set()


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
class _Thunk:
    __slots__ = ("fn", "done", "val")

    def __init__(self, fn):
        self.fn = fn
        self.done = False
        self.val = None

    def force(self):
        if not self.done:
            self.val = self.fn()
            self.done = True
            self.fn = None
        return self.val


class Evaluator:
    """Executes an `IRProgram`, charging the same meter as `cost.NativeEval`."""

    def __init__(self, registry, meter=None):
        self.registry = registry
        self.meter = meter or _cost.Meter()
        self.native = _cost.NativeEval(registry, self.meter)

    def run(self, prog, inputs):
        env = {}
        for k, v in inputs.items():
            env[k] = _Thunk(lambda v=v: v)
        for name, expr in prog.bindings:
            env[name] = _Thunk(lambda e=expr, env=env: self.eval(e, env))
        return self.eval(prog.output, env)

    def eval(self, e, env):
        t = type(e)
        if t is Lit:
            return e.value
        if t is Ref:
            v = env[e.name]
            return v.force() if isinstance(v, _Thunk) else v
        if t is Var:
            v = env[e.name]
            return v.force() if isinstance(v, _Thunk) else v
        if t is Prim:
            return self.native.prim(e.op, [self.eval(a, env) for a in e.args])
        if t is Call:
            return self.native.call_module(e.module, [self.eval(a, env) for a in e.args])
        if t is Guard:
            self.meter.hit("guard")
            c = self.eval(e.cond, env)
            return self.eval(e.then if c else e.other, env)
        if t is Let:
            env2 = dict(env)
            env2[e.name] = _Thunk(lambda: self.eval(e.value, env))
            return self.eval(e.body, env2)
        if t is Find:
            for i in range(e.lo, e.hi):
                env2 = dict(env)
                env2[e.var] = i
                self.meter.hit("find_test")
                if self.eval(e.pred, env2):
                    return self.eval(e.then, env2)
            return self.eval(e.default, env)
        if t is Scan:
            acc = self.eval(e.init, env)
            for i in range(e.lo, e.hi):
                env2 = dict(env)
                env2[e.var] = i
                env2[e.acc] = acc
                self.meter.hit("scan_step")
                if e.while_ is not None and not self.eval(e.while_, env2):
                    break
                acc = self.eval(e.step, env2)
            return acc
        raise TypeError(type(e))


# ---------------------------------------------------------------------------
# lowering the specification into the IR, one binding per node
# ---------------------------------------------------------------------------
def from_program(program, registry):
    """A 1:1 lazy transcription of a frozen `Program`.  No construct is added.

    Every `Prim` argument is strict, so this transcription forces exactly the
    same nodes the eager interpreter does.  It is the honest starting point of
    the search, and the control that shows laziness of *binding* is not the win.
    """
    from tcn.types import decode
    bindings = []
    for k, v in program.constants:
        bindings.append((k, Lit(decode(v.type, v.raw), k)))
    for node in program.nodes:
        c = node.candidates[node.selected]
        args = tuple(Ref(s) for s in c.sources)
        e = Call(c.operator.name, args) if c.operator.name.startswith("module:") else Prim(c.operator, args)
        bindings.append((node.name, e))
    out = program.outputs
    assert len(out) == 1, "miniature has one output"
    return IRProgram(bindings, Ref(out[0][1]), inputs=tuple(k for k, _ in program.inputs))


# ---------------------------------------------------------------------------
# lowering the IR to standalone stdlib-only Python
# ---------------------------------------------------------------------------
_PY_BIN = {"add": "+", "sub": "-", "mul": "*", "mod": "%", "idiv": "//"}
_PY_CMP = {"eq": "==", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}


class Lowerer:
    """Generate standalone Python.  stdlib only; no `tcn` import at run time."""

    def __init__(self, registry):
        self.registry = registry
        self.helpers = []
        self.modules = {}
        self.n = 0

    def fresh(self, p="v"):
        self.n += 1
        return "%s%d" % (p, self.n)

    def _rng(self, t):
        M = 2 ** t.bits
        lo = -(M // 2) if t.encoding.signed else 0
        hi = M // (2 if t.encoding.signed else 1) - 1
        return lo, hi

    def _chk(self, t, expr):
        if t.kind == "bool":
            return expr
        lo, hi = self._rng(t)
        return "_ck(%s, %d, %d)" % (expr, lo, hi)

    def module_fn(self, name):
        if name in self.modules:
            return self.modules[name]
        m = self.registry.modules[name]
        fn = "_m%d" % len(self.modules)
        self.modules[name] = fn
        from tcn.types import decode
        lines = ["def %s(%s):" % (fn, ", ".join(k for k, _ in m.inputs))]
        env = {k: k for k, _ in m.inputs}
        for k, v in m.constants:
            env[k] = repr(decode(v.type, v.raw))
        for node in m.nodes:
            c = node.candidates[node.selected]
            dst = self.fresh("t")
            lines.append("    %s = %s" % (dst, self.prim_expr(c.operator, [env[s] for s in c.sources])))
            env[node.name] = dst
        lines.append("    return %s" % env[m.outputs[0][1]])
        lines.append("")
        self.helpers.append("\n".join(lines))
        return fn

    def prim_expr(self, op, a):
        n = op.name
        p = dict(op.parameters)
        if n.startswith("module:"):
            return "%s(%s)" % (self.module_fn(n), ", ".join(a))
        if n == "project":
            return "%s[%d]" % (a[0], p["index"])
        if n == "index":
            return "_ix(%s, %s)" % (a[0], a[1])
        if n == "tuple":
            return "(%s,)" % ", ".join(a)
        if n == "identity":
            return a[0]
        if n in _PY_CMP:
            return "(%s %s %s)" % (a[0], _PY_CMP[n], a[1])
        if n == "not":
            return "(not %s)" % a[0]
        if n == "mux":
            return "(%s if %s else %s)" % (a[1], a[0], a[2])
        if n in _PY_BIN:
            if n in ("mod", "idiv"):
                return self._chk(op.output, "_dz(%s, %s)" % (a[0], a[1]) if n == "mod" else "_dzi(%s, %s)" % (a[0], a[1]))
            return self._chk(op.output, "(%s %s %s)" % (a[0], _PY_BIN[n], a[1]))
        if n in ("min", "max"):
            return self._chk(op.output, "%s(%s, %s)" % (n, a[0], a[1]))
        if n == "encode":
            if op.output.kind == "bool":
                return "(%s >= %r)" % (a[0], p.get("threshold", .5))
            return self._chk(op.output, "round(float(%s))" % a[0])
        if n == "insert":
            return "_ins(%s, %s, %d)" % (a[0], a[1], op.output.capacity)
        if n == "remove":
            return "(%s - {%s})" % (a[0], a[1])
        if n == "member":
            return "(%s in %s)" % (a[1], a[0])
        if n == "union":
            return "_cap(%s | %s, %d)" % (a[0], a[1], op.output.capacity)
        if n == "intersection":
            return "(%s & %s)" % (a[0], a[1])
        if n == "count":
            return "len(%s)" % a[0]
        if n == "not":
            return "(not %s)" % a[0]
        if n in ("and", "or", "xor", "nand", "nor", "xnor"):
            return {"and": "(%s and %s)", "or": "(%s or %s)", "xor": "(%s != %s)",
                    "nand": "(not (%s and %s))", "nor": "(not (%s or %s))",
                    "xnor": "(%s == %s)"}[n] % (a[0], a[1])
        if n.startswith("truth_"):
            k = int(n[6:])
            table = tuple(bool((k >> j) & 1) for j in range(4))
            return "%r[2 * %s + %s]" % (table, a[0], a[1])
        if n == "sum":
            return self._chk(op.output, "sum(%s)" % a[0])
        if n in ("reduce_min", "reduce_max"):
            return "%s(%s)" % ("min" if n == "reduce_min" else "max", a[0])
        if n == "pack":
            parts, off = [], 0
            for k, it in enumerate(op.inputs[0].items):
                parts.append("(int(%s[%d]) << %d)" % (a[0], k, off))
                off += 1 if it.kind == "bool" else it.bits
            return self._chk(op.output, "(%s)" % " | ".join(parts) if parts else "0")
        raise KeyError("cannot lower operator: %s" % n)

    # -- demand-driven lowering -------------------------------------------
    def emit(self, e, scope):
        """`(lines, varname)`.  A binding is emitted where it is first demanded.

        `scope` maps a binding name to the variable already holding its value
        *on the current path*.  Entering a `Guard` branch, a `Find` body or a
        `Scan` body copies the scope, so a value computed inside a branch is
        never assumed available outside it and a value computed in a loop body
        is recomputed each iteration.  That is what makes the generated code's
        executed-operation count equal the IR evaluator's, which is the property
        criterion 5 rests on.  Hoisting a binding above the `Guard` that guards
        it is exactly the mistake `tcn/compile.py` cannot avoid (DESIGN sec 0.2).
        """
        t = type(e)
        if t is Lit:
            v = self.fresh("k")
            return ["%s = %r" % (v, e.value)], v
        if t is Var:
            return [], scope[e.name]
        if t is Ref:
            if e.name in scope:
                return [], scope[e.name]
            if e.name not in self.table:
                raise KeyError("unbound %s" % e.name)
            lines, v = self.emit(self.table[e.name], scope)
            scope[e.name] = v
            return lines, v
        if t in (Prim, Call):
            lines, names = [], []
            for a in e.args:
                ls, nm = self.emit(a, scope)
                lines += ls
                names.append(nm)
            out = self.fresh()
            if t is Prim:
                lines.append("%s = %s" % (out, self.prim_expr(e.op, names)))
            else:
                lines.append("%s = %s(%s)" % (out, self.module_fn(e.module), ", ".join(names)))
            return lines, out
        if t is Guard:
            lines, c = self.emit(e.cond, scope)
            out = self.fresh("g")
            # A binding both branches necessarily force is forced on every path,
            # so emitting it once ahead of the branch changes no executed
            # operation and keeps the generated code linear instead of
            # exponential in the number of nested guards.
            common = necessary(e.then, self.table) & necessary(e.other, self.table)
            for nm, _ in self.bind_order:
                if nm in common and nm not in scope:
                    ls, _v = self.emit(Ref(nm), scope)
                    lines += ls
            ta, va = self.emit(e.then, dict(scope))
            fb, vb = self.emit(e.other, dict(scope))
            lines.append("if %s:" % c)
            lines += ["    " + s for s in ta] + ["    %s = %s" % (out, va)]
            lines.append("else:")
            lines += ["    " + s for s in fb] + ["    %s = %s" % (out, vb)]
            return lines, out
        if t is Let:
            lines, v = self.emit(e.value, scope)
            s2 = dict(scope)
            s2[e.name] = v
            ls, out = self.emit(e.body, s2)
            return lines + ls, out
        if t is Find:
            iv = self.fresh("i")
            out = self.fresh("f")
            s2 = dict(scope)
            s2[e.var] = iv
            pl, pv = self.emit(e.pred, s2)
            s3 = dict(s2)
            tl, tv = self.emit(e.then, s3)
            dl, dv = self.emit(e.default, dict(scope))
            lines = ["for %s in range(%d, %d):" % (iv, e.lo, e.hi)]
            lines += ["    " + s for s in pl]
            lines.append("    if %s:" % pv)
            lines += ["        " + s for s in tl]
            lines.append("        %s = %s" % (out, tv))
            lines.append("        break")
            lines.append("else:")
            lines += ["    " + s for s in dl]
            lines.append("    %s = %s" % (out, dv))
            return lines, out
        if t is Scan:
            iv = self.fresh("i")
            acc = self.fresh("a")
            lines, iv0 = self.emit(e.init, scope)
            lines.append("%s = %s" % (acc, iv0))
            s2 = dict(scope)
            s2[e.var] = iv
            s2[e.acc] = acc
            body = []
            if e.while_ is not None:
                wl, wv = self.emit(e.while_, dict(s2))
                body += wl + ["if not %s: break" % wv]
            sl, sv = self.emit(e.step, dict(s2))
            body += sl + ["%s = %s" % (acc, sv)]
            lines.append("for %s in range(%d, %d):" % (iv, e.lo, e.hi))
            lines += ["    " + s for s in body]
            return lines, acc
        raise TypeError(type(e))

    def program(self, prog, entry="run"):
        self.table = dict(prog.bindings)
        self.bind_order = list(prog.bindings)
        scope = {k: k for k in prog.inputs}
        lines, v = self.emit(prog.output, scope)
        src = [_PRELUDE, ""]
        src += self.helpers
        src.append("def %s(%s):" % (entry, ", ".join(prog.inputs)))
        src += ["    " + s for s in lines]
        src.append("    return %s" % v)
        src.append("")
        return "\n".join(src)


_PRELUDE = '''"""Generated by research/lazy-guard.  Standard library only."""


def _ck(x, lo, hi):
    if not lo <= x <= hi:
        raise OverflowError("value outside integer range")
    return x


def _dz(a, b):
    if b == 0:
        raise ValueError("zero denominator")
    return a % b


def _dzi(a, b):
    if b == 0:
        raise ValueError("zero denominator")
    return a // b


def _ix(t, i):
    if not 0 <= i < len(t):
        raise IndexError("index outside tuple")
    return t[i]


def _ins(s, x, cap):
    out = s | {x}
    if len(out) > cap:
        raise OverflowError("set capacity exceeded")
    return out


def _cap(s, cap):
    if len(s) > cap:
        raise OverflowError("set capacity exceeded")
    return s
'''


def load(source, name="lazyguard_generated"):
    import types as _t
    m = _t.ModuleType(name)
    m.__dict__["__file__"] = "<%s>" % name
    exec(compile(source, "<%s>" % name, "exec"), m.__dict__)
    return m
