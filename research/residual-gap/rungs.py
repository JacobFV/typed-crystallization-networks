"""The ladder: eight rungs from arm B to arm C, each a named transform.

PREREGISTRATION.md section 3.  R1 and R2 are produced **mechanically** from the
committed `research/lazy-guard/out/stage3_algorithm.py` so there is no
transcription risk on the largest rewrite; R3-R6 are written out and are proved
correct by the exhaustive gate in `gate.py`, not by inspection.  R0 is arm B
imported verbatim and R7 is arm C verbatim, both from
`research/lazy-latency/arms.py`.

Nothing under `tcn/`, `generators/`, `research/lazy-guard/`,
`research/lazy-latency/` or `research/compiled-runtime/` is modified.
"""
from __future__ import annotations

import ast
import pathlib
import sys
import types

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "research" / "lazy-latency"),
           str(ROOT / "research" / "compiled-runtime")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)

OVF = 'raise OverflowError("value outside integer range")'
IDX = 'raise IndexError("index outside tuple")'
DIV = 'raise ValueError("zero denominator")'


# ---------------------------------------------------------------------------
# R1 -- guard-call inlining, done mechanically on arm B's own source
# ---------------------------------------------------------------------------
def inline_guards(src: str) -> str:
    """Replace every `_ck` / `_ix` / `_dz` / `_dzi` call with inline statements.

    Every check the helper performed is still performed, in the same order, and
    raises the same exception with the same message.  Only the Python call
    frame goes away.  The transform is driven by `ast` rather than by regex so
    the nested `_ck(_dzi(a, b), lo, hi)` forms are handled exactly.
    """
    out = []
    for raw in src.splitlines():
        stripped = raw.strip()
        indent = raw[:len(raw) - len(raw.lstrip())]
        if not stripped or not any(h in stripped for h in ("_ck(", "_ix(", "_dz(", "_dzi(")):
            out.append(raw)
            continue
        try:
            node = ast.parse(stripped).body[0]
        except SyntaxError:                                   # pragma: no cover
            out.append(raw)
            continue
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            out.append(raw)
            continue
        dest = node.targets[0].id
        out.extend(indent + ln for ln in _expand(dest, node.value))
    return "\n".join(out) + "\n"


def _expand(dest, expr):
    """Statements that assign `expr` to `dest` with the helpers inlined."""
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
        fn = expr.func.id
        if fn == "_ck":
            inner, lo, hi = expr.args
            lines = _expand(dest, inner)
            lines.append("if not %s <= %s <= %s: %s"
                         % (ast.unparse(lo), dest, ast.unparse(hi), OVF))
            return lines
        if fn in ("_dz", "_dzi"):
            a, b = ast.unparse(expr.args[0]), ast.unparse(expr.args[1])
            op = "%" if fn == "_dz" else "//"
            return ["if %s == 0: %s" % (b, DIV),
                    "%s = %s %s %s" % (dest, a, op, b)]
        if fn == "_ix":
            t, i = ast.unparse(expr.args[0]), ast.unparse(expr.args[1])
            return ["if not 0 <= %s < len(%s): %s" % (i, t, IDX),
                    "%s = %s[%s]" % (dest, t, i)]
    return ["%s = %s" % (dest, ast.unparse(expr))]


def strip_helpers(src: str) -> str:
    """Drop the now-unused `_ck`/`_ix`/`_dz`/`_dzi`/`_ins`/`_cap` definitions."""
    tree = ast.parse(src)
    keep = [n for n in tree.body
            if not (isinstance(n, ast.FunctionDef)
                    and n.name in ("_ck", "_ix", "_dz", "_dzi", "_ins", "_cap"))]
    tree.body = keep
    return ast.unparse(tree) + "\n"


# ---------------------------------------------------------------------------
# R2 -- module-call inlining, also mechanical
# ---------------------------------------------------------------------------
def inline_module(src: str, fname="_m0") -> str:
    """Splice `_m0`'s body into each `X = _m0(a, b, obs)` call site.

    Formal parameters are substituted by the actual argument names (all three
    are plain names at every call site, so substitution is capture-free), and
    the body's locals are given a per-site prefix so two call sites cannot
    collide.
    """
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fname)
    params = [a.arg for a in fn.args.args]
    body_src = [ast.unparse(s) for s in fn.body[:-1]]
    ret = fn.body[-1]
    assert isinstance(ret, ast.Return) and isinstance(ret.value, ast.Name)
    ret_name = ret.value.id
    locals_ = sorted({t.id for s in fn.body for t in ast.walk(s)
                      if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)
                      and t.id not in params})

    out, site = [], 0
    for raw in src.splitlines():
        stripped = raw.strip()
        indent = raw[:len(raw) - len(raw.lstrip())]
        if ("= " + fname + "(") not in stripped:
            out.append(raw)
            continue
        node = ast.parse(stripped).body[0]
        dest = node.targets[0].id
        actuals = [ast.unparse(a) for a in node.value.args]
        site += 1
        ren = {p: a for p, a in zip(params, actuals)}
        ren.update({v: "s%d_%s" % (site, v) for v in locals_})
        out.append(indent + "# inlined %s(%s)" % (fname, ", ".join(actuals)))
        for line in body_src:
            for sub in _rename(line, ren).splitlines():
                out.append(indent + sub)
        out.append(indent + "%s = %s" % (dest, ren[ret_name]))
    txt = "\n".join(out) + "\n"
    tree = ast.parse(txt)
    tree.body = [n for n in tree.body
                 if not (isinstance(n, ast.FunctionDef) and n.name == fname)]
    return ast.unparse(tree) + "\n"


class _Ren(ast.NodeTransformer):
    def __init__(self, mapping):
        self.m = mapping

    def visit_Name(self, node):
        if node.id in self.m:
            return ast.parse(self.m[node.id], mode="eval").body
        return node


def _rename(line, mapping):
    tree = ast.parse(line)
    tree = _Ren(mapping).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


# ---------------------------------------------------------------------------
# R3 -- loop-invariant code motion and common-subexpression elimination
# ---------------------------------------------------------------------------
R3_SRC = '''\
def run(rec):
    pos = rec[0]
    obs = rec[1]
    n = len(obs)
    if 3 == 0: %(DIV)s
    v3 = pos // 3
    if not 0 <= v3 <= 65535: %(OVF)s
    if 32 == 0: %(DIV)s
    v5 = v3 %% 32
    if not 0 <= v5 <= 65535: %(OVF)s
    if 32 == 0: %(DIV)s
    v6 = v3 // 32
    if not 0 <= v6 <= 65535: %(OVF)s
    # hoisted out of both loops: the anchor pixel's three bytes, which the
    # module recomputed on every iteration of both loops
    a1 = pos + 1
    if not 0 <= a1 <= 65535: %(OVF)s
    a2 = pos + 2
    if not 0 <= a2 <= 65535: %(OVF)s
    if not 0 <= pos < n: %(IDX)s
    c0 = obs[pos]
    if not 0 <= a1 < n: %(IDX)s
    c1 = obs[a1]
    if not 0 <= a2 < n: %(IDX)s
    c2 = obs[a2]
    for i7 in range(1, 32):
        v12 = 3 * i7
        if not 0 <= v12 <= 65535: %(OVF)s
        v13 = pos + v12
        if not 0 <= v13 <= 65535: %(OVF)s
        v15 = min(v13, 3069)
        if not 0 <= v15 <= 65535: %(OVF)s
        t19 = v15 + 1
        if not 0 <= t19 <= 65535: %(OVF)s
        t20 = v15 + 2
        if not 0 <= t20 <= 65535: %(OVF)s
        if not 0 <= v15 < n: %(IDX)s
        t21 = obs[v15]
        if not 0 <= t19 < n: %(IDX)s
        t22 = obs[t19]
        if not 0 <= t20 < n: %(IDX)s
        t23 = obs[t20]
        t29 = t21 == c0
        t30 = t22 == c1
        t31 = t23 == c2
        t32 = (True, True, True, False)[2 * t29 + t30]
        v18 = (False, True, False, False)[2 * t32 + t31]
        v37 = v5 + i7
        if not 0 <= v37 <= 65535: %(OVF)s
        v38 = v37 < 32
        if not (v18 and v38):
            f8 = i7
            break
    else:
        f8 = 1
    for i42 in range(1, 32):
        v47 = 96 * i42
        if not 0 <= v47 <= 65535: %(OVF)s
        v48 = pos + v47
        if not 0 <= v48 <= 65535: %(OVF)s
        v50 = min(v48, 3069)
        if not 0 <= v50 <= 65535: %(OVF)s
        u19 = v50 + 1
        if not 0 <= u19 <= 65535: %(OVF)s
        u20 = v50 + 2
        if not 0 <= u20 <= 65535: %(OVF)s
        if not 0 <= v50 < n: %(IDX)s
        u21 = obs[v50]
        if not 0 <= u19 < n: %(IDX)s
        u22 = obs[u19]
        if not 0 <= u20 < n: %(IDX)s
        u23 = obs[u20]
        u29 = u21 == c0
        u30 = u22 == c1
        u31 = u23 == c2
        u32 = (True, True, True, False)[2 * u29 + u30]
        v53 = (False, True, False, False)[2 * u32 + u31]
        v57 = v6 + i42
        if not 0 <= v57 <= 65535: %(OVF)s
        v59 = v57 < 32
        if not (v53 and v59):
            f43 = i42
            break
    else:
        f43 = 1
    v72 = (int(c0) << 0) | (int(c1) << 8) | (int(c2) << 16)
    if not 0 <= v72 <= 16777215: %(OVF)s
    v73 = pos - 3
    if not 0 <= v73 <= 65535: %(OVF)s
    if not 0 <= v73 < n: %(IDX)s
    v74 = obs[v73]
    v75 = v73 + 1
    if not 0 <= v75 <= 65535: %(OVF)s
    if not 0 <= v75 < n: %(IDX)s
    v76 = obs[v75]
    v77 = v73 + 2
    if not 0 <= v77 <= 65535: %(OVF)s
    if not 0 <= v77 < n: %(IDX)s
    v78 = obs[v77]
    v80 = (int(v74) << 0) | (int(v76) << 8) | (int(v78) << 16)
    if not 0 <= v80 <= 16777215: %(OVF)s
    return (v5, v6, f8, f43, v72, v80)
''' % {"DIV": DIV, "OVF": OVF, "IDX": IDX}


# ---------------------------------------------------------------------------
# R4 -- short-circuiting the learned truth-table conjunction
# ---------------------------------------------------------------------------
# `_m0` decides equality of three bytes with two `truth_7` table lookups:
#   t32 = (True, True, True, False)[2*t29 + t30]     ==  not (t29 and t30)
#   t33 = (False, True, False, False)[2*t32 + t31]   ==  (not t32) and t31
# so t33 == t29 and t30 and t31.  A table lookup cannot short-circuit; `and`
# can, and then the 2nd and 3rd bytes are never loaded once the 1st differs.
# This CHANGES the executed operation count, so it is classed structural.
R4_SRC = '''\
def run(rec):
    pos = rec[0]
    obs = rec[1]
    n = len(obs)
    if 3 == 0: %(DIV)s
    v3 = pos // 3
    if not 0 <= v3 <= 65535: %(OVF)s
    if 32 == 0: %(DIV)s
    v5 = v3 %% 32
    if not 0 <= v5 <= 65535: %(OVF)s
    if 32 == 0: %(DIV)s
    v6 = v3 // 32
    if not 0 <= v6 <= 65535: %(OVF)s
    a1 = pos + 1
    if not 0 <= a1 <= 65535: %(OVF)s
    a2 = pos + 2
    if not 0 <= a2 <= 65535: %(OVF)s
    if not 0 <= pos < n: %(IDX)s
    c0 = obs[pos]
    if not 0 <= a1 < n: %(IDX)s
    c1 = obs[a1]
    if not 0 <= a2 < n: %(IDX)s
    c2 = obs[a2]
    for i7 in range(1, 32):
        v12 = 3 * i7
        if not 0 <= v12 <= 65535: %(OVF)s
        v13 = pos + v12
        if not 0 <= v13 <= 65535: %(OVF)s
        v15 = min(v13, 3069)
        if not 0 <= v15 <= 65535: %(OVF)s
        if not 0 <= v15 < n: %(IDX)s
        v18 = obs[v15] == c0
        if v18:
            t19 = v15 + 1
            if not 0 <= t19 <= 65535: %(OVF)s
            if not 0 <= t19 < n: %(IDX)s
            v18 = obs[t19] == c1
            if v18:
                t20 = v15 + 2
                if not 0 <= t20 <= 65535: %(OVF)s
                if not 0 <= t20 < n: %(IDX)s
                v18 = obs[t20] == c2
        v37 = v5 + i7
        if not 0 <= v37 <= 65535: %(OVF)s
        v38 = v37 < 32
        if not (v18 and v38):
            f8 = i7
            break
    else:
        f8 = 1
    for i42 in range(1, 32):
        v47 = 96 * i42
        if not 0 <= v47 <= 65535: %(OVF)s
        v48 = pos + v47
        if not 0 <= v48 <= 65535: %(OVF)s
        v50 = min(v48, 3069)
        if not 0 <= v50 <= 65535: %(OVF)s
        if not 0 <= v50 < n: %(IDX)s
        v53 = obs[v50] == c0
        if v53:
            u19 = v50 + 1
            if not 0 <= u19 <= 65535: %(OVF)s
            if not 0 <= u19 < n: %(IDX)s
            v53 = obs[u19] == c1
            if v53:
                u20 = v50 + 2
                if not 0 <= u20 <= 65535: %(OVF)s
                if not 0 <= u20 < n: %(IDX)s
                v53 = obs[u20] == c2
        v57 = v6 + i42
        if not 0 <= v57 <= 65535: %(OVF)s
        v59 = v57 < 32
        if not (v53 and v59):
            f43 = i42
            break
    else:
        f43 = 1
    v72 = (int(c0) << 0) | (int(c1) << 8) | (int(c2) << 16)
    if not 0 <= v72 <= 16777215: %(OVF)s
    v73 = pos - 3
    if not 0 <= v73 <= 65535: %(OVF)s
    if not 0 <= v73 < n: %(IDX)s
    v74 = obs[v73]
    v75 = v73 + 1
    if not 0 <= v75 <= 65535: %(OVF)s
    if not 0 <= v75 < n: %(IDX)s
    v76 = obs[v75]
    v77 = v73 + 2
    if not 0 <= v77 <= 65535: %(OVF)s
    if not 0 <= v77 < n: %(IDX)s
    v78 = obs[v77]
    v80 = (int(v74) << 0) | (int(v76) << 8) | (int(v78) << 16)
    if not 0 <= v80 <= 16777215: %(OVF)s
    return (v5, v6, f8, f43, v72, v80)
''' % {"DIV": DIV, "OVF": OVF, "IDX": IDX}


# ---------------------------------------------------------------------------
# R5a -- typed-guard elimination INCLUDING the `min(., 3069)` index clamp.
#        Pre-registered as the rung most at risk.  It is expected to be timed
#        only if it passes the gate.
# ---------------------------------------------------------------------------
R5A_SRC = '''\
def run(rec):
    pos = rec[0]
    obs = rec[1]
    v3 = pos // 3
    v5 = v3 % 32
    v6 = v3 // 32
    c0 = obs[pos]
    c1 = obs[pos + 1]
    c2 = obs[pos + 2]
    for i7 in range(1, 32):
        v15 = pos + 3 * i7
        v18 = obs[v15] == c0 and obs[v15 + 1] == c1 and obs[v15 + 2] == c2
        v38 = v5 + i7 < 32
        if not (v18 and v38):
            f8 = i7
            break
    else:
        f8 = 1
    for i42 in range(1, 32):
        v50 = pos + 96 * i42
        v53 = obs[v50] == c0 and obs[v50 + 1] == c1 and obs[v50 + 2] == c2
        v59 = v6 + i42 < 32
        if not (v53 and v59):
            f43 = i42
            break
    else:
        f43 = 1
    v72 = (int(c0) << 0) | (int(c1) << 8) | (int(c2) << 16)
    v73 = pos - 3
    v80 = (int(obs[v73]) << 0) | (int(obs[v73 + 1]) << 8) | (int(obs[v73 + 2]) << 16)
    return (v5, v6, f8, f43, v72, v80)
'''


# ---------------------------------------------------------------------------
# R5 -- typed-guard elimination, clamp RETAINED
# ---------------------------------------------------------------------------
R5_SRC = '''\
def run(rec):
    pos = rec[0]
    obs = rec[1]
    v3 = pos // 3
    v5 = v3 % 32
    v6 = v3 // 32
    c0 = obs[pos]
    c1 = obs[pos + 1]
    c2 = obs[pos + 2]
    for i7 in range(1, 32):
        v15 = min(pos + 3 * i7, 3069)
        v18 = obs[v15] == c0 and obs[v15 + 1] == c1 and obs[v15 + 2] == c2
        v38 = v5 + i7 < 32
        if not (v18 and v38):
            f8 = i7
            break
    else:
        f8 = 1
    for i42 in range(1, 32):
        v50 = min(pos + 96 * i42, 3069)
        v53 = obs[v50] == c0 and obs[v50 + 1] == c1 and obs[v50 + 2] == c2
        v59 = v6 + i42 < 32
        if not (v53 and v59):
            f43 = i42
            break
    else:
        f43 = 1
    v72 = (int(c0) << 0) | (int(c1) << 8) | (int(c2) << 16)
    v73 = pos - 3
    v80 = (int(obs[v73]) << 0) | (int(obs[v73 + 1]) << 8) | (int(obs[v73 + 2]) << 16)
    return (v5, v6, f8, f43, v72, v80)
'''


# ---------------------------------------------------------------------------
# R6 -- loop-bound strength reduction: the bound test moves ahead of the pixel
#       test and into the loop form, which is what makes the clamp dead.
#       The `else: f8 = 1` arm is dropped; `research/lazy-guard` section 6.5 and
#       FINDINGS section 51 both record it as unreachable on the declared
#       domain, and arm C does not have it either.  The gate is what proves it.
# ---------------------------------------------------------------------------
R6_SRC = '''\
def run(rec):
    pos = rec[0]
    obs = rec[1]
    v3 = pos // 3
    v5 = v3 % 32
    v6 = v3 // 32
    c0 = obs[pos]
    c1 = obs[pos + 1]
    c2 = obs[pos + 2]
    w = 1
    while v5 + w < 32 and obs[pos + 3 * w] == c0 and obs[pos + 3 * w + 1] == c1 and obs[pos + 3 * w + 2] == c2:
        w += 1
    h = 1
    while v6 + h < 32 and obs[pos + 96 * h] == c0 and obs[pos + 96 * h + 1] == c1 and obs[pos + 96 * h + 2] == c2:
        h += 1
    v72 = (int(c0) << 0) | (int(c1) << 8) | (int(c2) << 16)
    v73 = pos - 3
    v80 = (int(obs[v73]) << 0) | (int(obs[v73 + 1]) << 8) | (int(obs[v73 + 2]) << 16)
    return (v5, v6, w, h, v72, v80)
'''


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
LABELS = {
    "R0": "arm B verbatim (research/lazy-guard/out/stage3_algorithm.py)",
    "R1": "+ guard-call inlining (_ck/_ix/_dz/_dzi)",
    "R2": "+ module-call inlining (_m0)",
    "R3": "+ loop-invariant code motion and CSE",
    "R4": "+ short-circuiting the truth-table conjunction",
    "R5a": "+ typed-guard elimination INCLUDING the min(.,3069) index clamp",
    "R5": "+ typed-guard elimination, clamp retained",
    "R6": "+ loop-bound strength reduction (bound test first, while form)",
    "R7": "arm C verbatim (research/compiled-runtime/fixtures.py reference)",
}
CLASS = {
    "R1": "representational", "R2": "representational", "R3": "representational",
    "R4": "structural", "R5": "representational", "R6": "representational",
    "R7": "unaccounted",
}
LADDER = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"]


def _module(name, src):
    mod = types.ModuleType("residual_%s" % name)
    mod.__dict__["__file__"] = str(OUT / ("%s.py" % name))
    exec(compile(src, str(OUT / ("%s.py" % name)), "exec"), mod.__dict__)
    return mod


def build():
    """Every rung as (name, source, module)."""
    import arms                                              # noqa: E402

    b_src = (ROOT / "research" / "lazy-guard" / "out" / "stage3_algorithm.py").read_text()
    r1 = strip_helpers(inline_guards(b_src))
    r2 = inline_module(r1)

    srcs = {"R0": b_src, "R1": r1, "R2": r2, "R3": R3_SRC, "R4": R4_SRC,
            "R5a": R5A_SRC, "R5": R5_SRC, "R6": R6_SRC,
            "R7": arms.REFERENCE_SOURCE}
    rungs = {}
    for k, s in srcs.items():
        (OUT / ("%s.py" % k)).write_text(s)
        rungs[k] = {"name": k, "label": LABELS[k], "source": s,
                    "class": CLASS.get(k, "baseline"),
                    "module": _module(k, s),
                    "path": str(OUT / ("%s.py" % k))}
    return rungs


if __name__ == "__main__":
    r = build()
    for k in LADDER + ["R5a"]:
        print("%-4s %-62s %6d bytes" % (k, r[k]["label"], len(r[k]["source"])))
