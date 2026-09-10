"""Where the guards are, per emitted function, and what each arm actually runs.

Section 59 attributed its 1.02x on `visual` at the opcode level rather than
arguing it, and found `_m1` -- 3,100 calls, 73.7 % of bytecodes -- untouched.
`_m1` is the whole question here, so the same instrument is pointed at it: guards
bucketed by the function that emits them, calls per screenshot, and executed
bytecodes by code object, for every arm.

The meters are `research/lazy-guard/cost.py`'s `sys.monitoring` counter and
`research/compiled-runtime/harness.py`'s call counter, reached through
`research/emitter-guards/attribute.py` -- extended, never re-implemented.
"""
from __future__ import annotations

import collections
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "research" / "compiled-runtime"),
           str(ROOT / "research" / "lazy-guard")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import harness                                          # noqa: E402
import arms as ARMS                                     # noqa: E402
from tcn import compile as C                            # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_EG = _load("eg_attribute", "research/emitter-guards/attribute.py")
by_code = _EG.by_code

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
ARMNAMES = ("A0", "A1", "B", "B+")


def guards_by_scope(program, registry, inline_bounded=False):
    """Which emitted function each kept / eliminated guard belongs to.

    Section 59's `attribute.py` version, extended with the G1b refinement-bound
    class this track adds; `AMENDMENT 5` there (excluding `module:`, `map` and
    `filter` from the counter-delta wrapper, so a module's inner guards are not
    charged to its call site as well) is preserved verbatim.
    """
    log = []
    real_scalar = C._Compiler._emit_scalar
    real_lower = C._Compiler._lower
    _scope = ["run"]

    def emit_scalar(self, dst, expr, op, s=()):
        before = {k: self.stats["guard_%s_eliminated" % k] for k in ("range", "bounds")}
        out = real_scalar(self, dst, expr, op, s)
        if self._fast_kind(op) == "int":
            log.append(("G1", self.stats["guard_range_eliminated"] == before["range"],
                        _scope[-1]))
            if op.output.bounds is not None:
                log.append(("G1b", self.stats["guard_bounds_eliminated"] == before["bounds"],
                            _scope[-1]))
        elif op.output.kind == "int" and op.output.bounds is not None:
            log.append(("Ccall", True, _scope[-1]))     # went to `_canon_fn` instead
        return out

    def lower(self, dst, op, s, scope):
        _scope.append(scope)
        b = {k: self.stats["guard_%s_emitted" % k] for k in ("index", "zerodiv", "shift")}
        try:
            out = real_lower(self, dst, op, s, scope)
        finally:
            _scope.pop()
        if not (op.name.startswith("module:") or op.name in {"map", "filter"}):
            for k, cls in (("index", "G2"), ("zerodiv", "G3"), ("shift", "G4")):
                if self.stats["guard_%s_emitted" % k] > b[k]:
                    log.append((cls, True, scope))
                elif op.name == "index" and k == "index":
                    log.append((cls, False, scope))
        return out

    C._Compiler._emit_scalar = emit_scalar
    C._Compiler._lower = lower
    try:
        res = C.compile_program(program, registry, inline_bounded=inline_bounded)
        fn_of = dict(res.stats.get("module_function", {}))
    finally:
        C._Compiler._emit_scalar = real_scalar
        C._Compiler._lower = real_lower
    kept, gone = collections.Counter(), collections.Counter()
    for cls, is_kept, scope in log:
        name = fn_of.get(scope, scope)
        (kept if is_kept else gone)["%s in %s" % (cls, name)] += 1
    return dict(sorted(kept.items())), dict(sorted(gone.items()))


def main(argv):
    names = [a for a in argv[1:]] or list(ARMNAMES)
    report = {}
    for arm in names:
        real, ib = ("B", True) if arm == "B+" else (arm, False)
        f, res, mod = ARMS.compiled(real, seeds=(200,), inline_bounded=True if ib else None)
        p, r = f["program"], f["registry"]
        case = {k: f["cases"][0][k] for k, _ in p.inputs}
        fns = [k for k in mod.__dict__ if k.startswith("_m")
               or (k.startswith("_c") and k != "_cexp" and callable(mod.__dict__[k]))]
        with harness.CallCounter(mod, fns) as cc:
            mod.run(dict(case), validate=False)
        calls = dict(cc.counts)
        buckets = dict(by_code(lambda m=mod: m.run(dict(case), validate=False)).most_common(14))
        kept, gone = guards_by_scope(p, r, inline_bounded=ib)
        total = sum(buckets.values())
        report[arm] = {"module_functions": res.stats.get("module_function", {}),
                       "calls_per_screenshot": calls,
                       "bytecodes_by_code_object": buckets,
                       "bytecodes_total_measured": total,
                       "guards_kept_by_function": kept,
                       "guards_eliminated_by_function": gone,
                       "guard_stats": {k: v for k, v in res.stats.items()
                                       if k.startswith("guard_")},
                       "source_bytes": res.stats["source_bytes"]}
        print("==", arm)
        print("   calls ", json.dumps(calls))
        print("   kept  ", json.dumps(kept))
        print("   gone  ", json.dumps(gone))
        print("   bc    ", json.dumps(buckets))
    (OUT / "attribute.json").write_text(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv)
