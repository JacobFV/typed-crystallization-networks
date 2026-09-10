"""Where the removed guards actually are, against where the time actually is.

`measure.py` records a 1.02x wall-clock ratio on `visual` where
FINDINGS section 56 measured 2.834x for the same transform on the `visual`
parser's S2 subroutine.  That is a surprising headline, so it is checked against
raw data here rather than explained away: executed bytecodes bucketed by code
object, and the call count of every emitted function, per case, for both arms.

The bytecode meter is `research/lazy-guard/cost.py`'s `sys.monitoring` counter
extended with a bucket key -- extended, not re-implemented -- exactly as
`research/residual-gap/counts.py` did.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "research" / "compiled-runtime"),
           str(ROOT / "research" / "lazy-guard")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cost as lazy_cost                               # noqa: E402
import fixtures                                        # noqa: E402
import gate as G                                       # noqa: E402
import harness                                         # noqa: E402
from tcn import compile as C                           # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
_M = lazy_cost._M
_TOOL = 5


def by_code(fn):
    """Executed bytecodes bucketed by the code object that executed them."""
    tally = collections.Counter()

    def on_instr(code, offset):
        tally[code.co_name] += 1
    _M.use_tool_id(_TOOL, "emitterguards")
    try:
        _M.register_callback(_TOOL, _M.events.INSTRUCTION, on_instr)
        _M.set_events(_TOOL, _M.events.INSTRUCTION)
        fn()
        _M.set_events(_TOOL, 0)
    finally:
        _M.register_callback(_TOOL, _M.events.INSTRUCTION, None)
        _M.free_tool_id(_TOOL)
    return tally


def guards_by_scope(program, registry):
    """Which emitted function each kept / eliminated guard belongs to."""
    log = []
    real_scalar = C._Compiler._emit_scalar
    real_lower = C._Compiler._lower
    fn_of = {}

    def emit_scalar(self, dst, expr, op, s=()):
        before = self.stats["guard_range_eliminated"]
        out = real_scalar(self, dst, expr, op, s)
        if self._fast_kind(op) == "int":
            log.append(("G1", self.stats["guard_range_eliminated"] == before, _scope[-1]))
        return out

    _scope = ["run"]

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
        return out

    C._Compiler._emit_scalar = emit_scalar
    C._Compiler._lower = lower
    try:
        res = C.compile_program(program, registry)
        fn_of = dict(res.stats.get("module_function", {}))
    finally:
        C._Compiler._emit_scalar = real_scalar
        C._Compiler._lower = real_lower
    kept, gone = collections.Counter(), collections.Counter()
    for cls, is_kept, scope in log:
        name = fn_of.get(scope, scope)
        (kept if is_kept else gone)["%s in %s" % (cls, name)] += 1
    return dict(sorted(kept.items())), dict(sorted(gone.items()))


def subroutine_timing(mo, mn, case, fn_name, repeats=21):
    """The same guard elimination, timed on one emitted function in isolation.

    FINDINGS section 56 measured typed-guard elimination at 2.834x *on the S2
    subroutine*.  `measure.py` reports the whole-artifact number.  This is the
    bridge between them, on the same protocol: capture the arguments the
    artifact actually passes, then time both arms' function on exactly those.
    """
    import measure as M
    args = []
    real = getattr(mn, fn_name)

    def capture(*a):
        args.append(a)
        return real(*a)
    setattr(mn, fn_name, capture)
    try:
        mn.run(dict(case), validate=False)
    finally:
        setattr(mn, fn_name, real)
    if not args:
        return {"unavailable": "%s never called" % fn_name}
    calls = {"old": getattr(mo, fn_name), "new": getattr(mn, fn_name)}
    recs = list(args)
    while len(recs) < 256:
        recs += list(args)
    # bit-identity on exactly the arguments about to be timed
    bad = [a for a in args if calls["old"](*a) != calls["new"](*a)]
    if bad:
        return {"identical": False, "mismatches": len(bad)}
    for a in calls:
        M.sweep_ns(lambda x, f=calls[a]: f(*x), recs)
    samples = {a: [] for a in calls}
    for _ in range(repeats):
        for a in ("old", "new"):
            samples[a].append(M.sweep_ns(lambda x, f=calls[a]: f(*x), recs) / 1e3 / len(recs))
    bc = {a: lazy_cost.count_opcodes(lambda f=calls[a]: [f(*x) for x in args]) / len(args)
          for a in calls}
    return {"identical": True, "calls_timed": len(args), "sweep_calls": len(recs),
            "latency_us": {a: M.summarize(v) for a, v in samples.items()},
            "ratio_old_over_new": M.ratio_ci(samples["old"], samples["new"]),
            "bytecodes_per_call": bc,
            "bytecode_ratio_old_over_new": bc["old"] / bc["new"]}


def main(names):
    base_mod, _sha = G.baseline_compiler()
    report = {}
    for name in names:
        f = fixtures.FIXTURES[name]()
        p, r = f["program"], f["registry"]
        old = base_mod.compile_program(p, r)
        new = C.compile_program(p, r)
        mo, mn = old.module("o_" + name), new.module("n_" + name)
        case = {k: f["cases"][0][k] for k, _ in p.inputs}
        fns = [k for k in mn.__dict__ if k.startswith("_m")
               or (k.startswith("_c") and k != "_cexp" and callable(mn.__dict__[k]))]
        counts = {}
        for arm, m in (("old", mo), ("new", mn)):
            with harness.CallCounter(m, fns) as cc:
                m.run(dict(case), validate=False)
            counts[arm] = dict(cc.counts)
        buckets = {arm: dict(by_code(lambda m=m: m.run(dict(case), validate=False)).most_common(12))
                   for arm, m in (("old", mo), ("new", mn))}
        kept, gone = guards_by_scope(p, r)
        subs = {}
        for fn in sorted(set(fns)):
            if any(k.endswith("in %s" % fn) for k in gone):
                subs[fn] = subroutine_timing(mo, mn, case, fn)
        report[name] = {"subroutine_timing": subs,
                        "module_functions": new.stats.get("module_function", {}),
                        "calls_per_case": counts,
                        "bytecodes_by_code_object": buckets,
                        "guards_kept_by_function": kept,
                        "guards_eliminated_by_function": gone}
        print("==", name)
        print("   calls   ", json.dumps(counts["new"]))
        print("   gone    ", json.dumps(gone))
        print("   kept    ", json.dumps(kept))
        print("   bc old  ", json.dumps(buckets["old"]))
        print("   bc new  ", json.dumps(buckets["new"]))
        for fn, s in subs.items():
            if s.get("identical"):
                print("   sub %-4s %.4f -> %.4f us  x%.4f CI[%.4f, %.4f]  bytecodes %.1f -> %.1f (x%.3f)"
                      % (fn, s["latency_us"]["old"]["median_us"], s["latency_us"]["new"]["median_us"],
                         s["ratio_old_over_new"]["ratio"], s["ratio_old_over_new"]["ci95_lo"],
                         s["ratio_old_over_new"]["ci95_hi"], s["bytecodes_per_call"]["old"],
                         s["bytecodes_per_call"]["new"], s["bytecode_ratio_old_over_new"]))
            else:
                print("   sub %-4s %s" % (fn, json.dumps(s)))
    (OUT / "attribute.json").write_text(json.dumps(report, indent=1, default=str))
    print("-> %s" % (OUT / "attribute.json"))


if __name__ == "__main__":
    main(sys.argv[1:] or ["visual", "language"])
