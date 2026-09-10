"""PREREGISTRATION section 4 -- three cost measures on all four artifacts.

Arms: `old` = the emitter at `ee7d63c`, `new` = the emitter with typed-guard
elimination.  Same frozen `Program`, same `Registry`, same inputs.

Three measures, reported separately because FINDINGS section 51 established that
they disagree in both directions:

  * executed **primitives** -- leaf operator applications, worst and expected,
    metered by `research/lazy-guard/cost.py`'s `NativeEval`.  These are a
    property of the *program*, which this pass does not touch, so the registered
    prediction is exactly 1.00x and any other value is a bug (F5).
  * executed **CPython bytecodes** per case, worst and expected, counted with
    the same file's `count_opcodes`.
  * **wall clock**, batch-one warm, under `research/lazy-latency/latency.py`'s
    protocol verbatim -- one core, gc off around each sweep, a discarded warm-up
    sweep, 21 repeats with the two arms interleaved round-robin inside each
    repeat, medians with nonparametric 95% CIs and a 10,000-resample bootstrap
    CI on the ratio.

**No timing is recorded before this file has re-asserted bit-identity on the
exact cases it is about to time.**  That is separate from, and additional to,
`gate.py`'s much wider domain.
"""
from __future__ import annotations

import gc
import json
import math
import pathlib
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "research" / "compiled-runtime"),
           str(ROOT / "research" / "lazy-guard"), str(ROOT / "research" / "lazy-latency")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cost as lazy_cost                               # noqa: E402  the bytecode meter
import fixtures                                        # noqa: E402
import gate as G                                       # noqa: E402  outcome/canon/digest
from tcn.compile import compile_program                # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
REPEATS = 21
CPU = 19
BOOTSTRAP = 10000
MIN_SWEEP_NS = 2_000_000                               # 2 ms, so a sweep clears the clock


# ---------------------------------------------------------------------------
# statistics, taken verbatim from research/lazy-latency/latency.py
# ---------------------------------------------------------------------------
def median_ci(xs, alpha=0.05):
    n = len(xs)
    s = sorted(xs)
    lo, acc = 0, 0.0
    for k in range(n + 1):
        acc += math.comb(n, k) * 0.5 ** n
        if acc > alpha / 2:
            lo = k
            break
    hi = n - 1 - lo
    lo, hi = min(lo, hi), max(lo, hi)
    cover = sum(math.comb(n, k) * 0.5 ** n for k in range(lo, hi + 1))
    return s[lo], s[hi], cover


def summarize(xs):
    s = sorted(xs)
    lo, hi, cover = median_ci(s)
    return {"median_us": statistics.median(s), "min_us": s[0], "max_us": s[-1],
            "ci95_lo_us": lo, "ci95_hi_us": hi, "ci_coverage": round(cover, 4),
            "repeats": len(s)}


def ratio_ci(num, den, seed=20260910, resamples=BOOTSTRAP):
    """Bootstrap 95% CI of median(num)/median(den) over paired repeats."""
    import random
    rng = random.Random(seed)
    n = min(len(num), len(den))
    pt = statistics.median(num) / statistics.median(den)
    boot = []
    for _ in range(resamples):
        pick = [rng.randrange(n) for _ in range(n)]
        boot.append(statistics.median([num[i] for i in pick]) /
                    statistics.median([den[i] for i in pick]))
    boot.sort()
    return {"ratio": pt, "ci95_lo": boot[int(0.025 * resamples)],
            "ci95_hi": boot[int(0.975 * resamples)],
            "contains_1": boot[int(0.025 * resamples)] <= 1.0 <= boot[int(0.975 * resamples)]}


def pin(cpu=CPU):
    import os
    try:
        os.sched_setaffinity(0, {cpu})
        return sorted(os.sched_getaffinity(0))
    except OSError as exc:
        return "unavailable: %s" % exc


def loadavg():
    return pathlib.Path("/proc/loadavg").read_text().split()[:3]


def sweep_ns(call, recs):
    gc.collect()
    gc.disable()
    try:
        t = time.perf_counter_ns()
        for r in recs:
            call(r)
        return time.perf_counter_ns() - t
    finally:
        gc.enable()


# ---------------------------------------------------------------------------
def build_arms(name):
    base_mod, base_sha = G.baseline_compiler()
    f = fixtures.FIXTURES[name]()
    p, r = f["program"], f["registry"]
    old = base_mod.compile_program(p, r)
    new = compile_program(p, r)
    cases = [{k: c[k] for k, _ in p.inputs} for c in f["cases"]]
    return f, p, r, {"old": (old, old.module("old_" + name)),
                     "new": (new, new.module("new_" + name))}, cases, base_sha


def identity_before_timing(mods, cases):
    """PREREGISTRATION section 3 gate A, re-asserted on the timed cases themselves."""
    rows = {}
    for arm, (_res, m) in mods.items():
        rows[arm] = [G.outcome(lambda c=c, m=m, v=v: m.run(dict(c), validate=v))
                     for c in cases for v in (True, False)]
    ok = rows["old"] == rows["new"]
    return {"identical": ok, "comparisons": len(rows["old"]),
            "digest_old": G.digest(rows["old"]), "digest_new": G.digest(rows["new"])}


def primitives(program, registry, cases):
    """Executed leaf operators, worst and expected -- a property of the *program*.

    Two instruments, because `research/lazy-guard/cost.py`'s `NativeEval` is a
    miniature evaluator that has not been taught `sin` or `decode` and so cannot
    run `mixed` or `computer`.  The portable one meters the shipped interpreter
    itself: one `Registry.exact` application per leaf operator, module
    invocations excluded from the count exactly as `cost.py` defines it (a
    module call is not a primitive; the leaves inside it are).  Where both run,
    they are reported together and must agree.

    Neither arm appears here.  The emitter does not change the program, so this
    measure is arm-independent by construction; F5 makes any other outcome a bug.
    """
    from tcn.types import Value
    types = dict(program.inputs)
    real = registry.exact
    tally = {"n": 0}

    def counted(op, args):
        if not op.name.startswith("module:"):
            tally["n"] += 1
        return real(op, args)

    per_case = []
    registry.exact = counted
    try:
        for c in cases:
            tally["n"] = 0
            program.run({k: Value.of(types[k], c[k]) for k in types}, registry=registry)
            per_case.append(tally["n"])
    finally:
        registry.exact = real
    rep = {"instrument": "Registry.exact applications, module calls excluded",
           "cases": len(per_case), "worst_ops": max(per_case),
           "best_ops": min(per_case), "expected_ops": statistics.mean(per_case)}
    try:
        meter = lazy_cost.Meter()
        ev = lazy_cost.NativeEval(registry, meter)
        prof, _ = lazy_cost.profile(lambda c: ev.run(program, dict(c)), cases, meter)
        rep["native_eval"] = {k: prof[k] for k in ("worst_ops", "best_ops", "expected_ops")}
        rep["instruments_agree"] = (prof["worst_ops"] == rep["worst_ops"]
                                    and prof["expected_ops"] == rep["expected_ops"])
    except Exception as exc:
        rep["native_eval"] = "unavailable: %s: %s" % (type(exc).__name__, exc)
    return rep


def bytecodes(mods, cases):
    out = {}
    for arm, (_res, m) in mods.items():
        counts = [lazy_cost.count_opcodes(lambda c=c, m=m: m.run(dict(c), validate=False))
                  for c in cases]
        out[arm] = {"expected": statistics.mean(counts), "worst": max(counts),
                    "best": min(counts), "cases": len(counts)}
    out["ratio_old_over_new"] = out["old"]["expected"] / out["new"]["expected"]
    return out


def timings(mods, cases, repeats=REPEATS):
    calls = {a: (lambda c, m=m: m.run(dict(c), validate=False)) for a, (_r, m) in mods.items()}
    # pad the sweep so it clears the clock, keeping every arm on the same multiset
    probe = max(1, sweep_ns(calls["new"], cases))
    mult = max(1, math.ceil(MIN_SWEEP_NS / probe))
    recs = list(cases) * mult
    for a in calls:                                    # warm-up, discarded
        calls[a](cases[0])
        sweep_ns(calls[a], recs)
    samples = {a: [] for a in calls}
    for _ in range(repeats):
        for a in ("old", "new"):                       # round-robin inside the repeat
            samples[a].append(sweep_ns(calls[a], recs) / 1e3 / len(recs))
    return ({a: summarize(v) for a, v in samples.items()},
            ratio_ci(samples["old"], samples["new"]),
            {"sweep_calls": len(recs), "case_multiplier": mult}), samples


def deployment(name, mods):
    """Bytes on disk and cold start, via research/compiled-runtime/deploy.py."""
    import deploy
    out_dir = OUT / "pyz"
    out_dir.mkdir(exist_ok=True)
    rep = {}
    for arm, (res, _m) in mods.items():
        path = deploy.build_c("%s_%s" % (name, arm), res, out_dir)
        r = deploy.run_pyz(path, "", 0, samples=5) if False else None
        colds = []
        rsss = []
        for _ in range(5):
            w, rss, _ = deploy._spawn(path, "")
            colds.append(w)
            rsss.append(rss)
        rep[arm] = {"source_bytes": len(res.source.encode()),
                    "source_lines": res.source.count("\n") + 1,
                    "pyz_bytes": path.stat().st_size,
                    "cold_start_ms": statistics.median(colds),
                    "cold_start_samples": sorted(colds),
                    "peak_rss_mb_cold": statistics.median(rsss)}
    return rep


def run(name, repeats=REPEATS):
    t0 = time.time()
    f, p, r, mods, cases, base_sha = build_arms(name)
    ident = identity_before_timing(mods, cases)
    rep = {"artifact": name, "what": f["what"], "cases": len(cases),
           "baseline_rev": G.BASE_REV, "baseline_source_sha256": base_sha,
           "identity_before_timing": ident,
           "guards": {a: {k: v for k, v in res.stats.items() if k.startswith("guard_")}
                      for a, (res, _m) in mods.items()},
           "program_digest": p.digest}
    import hashlib
    shas = {a: hashlib.sha256(res.source.encode()).hexdigest()
            for a, (res, _m) in mods.items()}
    rep["source_sha256"] = shas
    # An artifact whose two arms emit byte-identical source is a NULL CONTROL:
    # any measured wall-clock difference on it is the instrument's own bias.
    rep["null_control"] = shas["old"] == shas["new"]
    if not ident["identical"]:
        rep["timed"] = False
        rep["reason"] = "bit-identity failed on the timed cases; nothing timed"
        return rep
    rep["timed"] = True
    rep["primitives"] = primitives(p, r, cases)
    rep["bytecodes"] = bytecodes(mods, cases)
    rep["cpu_affinity"] = pin()
    rep["loadavg_before"] = loadavg()
    (lat, ratio, shape), _raw = timings(mods, cases, repeats)
    rep["latency_us"] = lat
    rep["latency_ratio_old_over_new"] = ratio
    rep["sweep"] = shape
    rep["loadavg_after"] = loadavg()
    rep["deployment"] = deployment(name, mods)
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


def main(argv):
    args = argv[1:]
    run_tag = args.pop(0) if args and args[0].startswith("run") else "run1"
    names = args or ["mixed", "language", "computer", "visual"]
    tag = "measure_%s" % run_tag if len(names) == 4 else "measure_%s_%s" % (run_tag, "_".join(names))
    report = {"repeats": REPEATS, "run": run_tag, "artifacts": {}}
    for n in names:
        print("== measure %s" % n, flush=True)
        rep = run(n)
        report["artifacts"][n] = rep
        if rep["timed"]:
            print("   identity %s (%d)  bytecodes %.0f -> %.0f  wall %.4f -> %.4f us  x%.4f %s"
                  % (rep["identity_before_timing"]["identical"],
                     rep["identity_before_timing"]["comparisons"],
                     rep["bytecodes"]["old"]["expected"], rep["bytecodes"]["new"]["expected"],
                     rep["latency_us"]["old"]["median_us"], rep["latency_us"]["new"]["median_us"],
                     rep["latency_ratio_old_over_new"]["ratio"],
                     "CI[%.4f, %.4f]" % (rep["latency_ratio_old_over_new"]["ci95_lo"],
                                         rep["latency_ratio_old_over_new"]["ci95_hi"])), flush=True)
        else:
            print("   NOT TIMED: %s" % rep["reason"], flush=True)
    path = OUT / ("%s.json" % tag)
    path.write_text(json.dumps(report, indent=1, default=str))
    print("-> %s" % path)


if __name__ == "__main__":
    main(sys.argv)
