"""PREREGISTRATION section 3 and 5 -- batch-one warm latency, three distributions.

Every number here is produced by the protocol fixed in `PREREGISTRATION.md`
before any of it was run: pinned to one core, gc collected then disabled around
each timed block, one discarded warm-up sweep per arm per distribution, 21
repeats, **arms interleaved round-robin inside each repeat** so drift is shared,
medians with a nonparametric 95% CI and a bootstrap CI on the ratio.

`research/compiled-runtime/harness.py` supplies the gc discipline and `rss_mb`;
it is imported rather than re-implemented.
"""
from __future__ import annotations

import gc
import json
import math
import pathlib
import random
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "research" / "compiled-runtime")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms                                          # noqa: E402
import equivalence                                   # noqa: E402
import harness                                       # noqa: E402  the shared instrument
import cost as lazy_cost                             # noqa: E402  bytecode meter

OUT = arms.OUT
REPEATS = 21
CPU = 19
BOOTSTRAP = 10000
ORDER = ("A", "B", "C")


# ---------------------------------------------------------------------------
# statistics fixed in the pre-registration
# ---------------------------------------------------------------------------
def median_ci(xs, alpha=0.05):
    """Nonparametric 95% CI of the median from order statistics."""
    n = len(xs)
    s = sorted(xs)
    lo = 0
    acc = 0.0
    for k in range(n + 1):
        acc += math.comb(n, k) * 0.5 ** n
        if acc > alpha / 2:
            lo = k
            break
    hi = n - 1 - lo
    lo = min(lo, hi)
    hi = max(lo, hi)
    cover = sum(math.comb(n, k) * 0.5 ** n for k in range(lo, hi + 1))
    return s[lo], s[hi], cover


def summarize(xs):
    s = sorted(xs)
    q = statistics.quantiles(s, n=4, method="inclusive") if len(s) >= 4 else [s[0]] * 3
    lo, hi, cover = median_ci(s)
    return {"median_us": statistics.median(s), "min_us": s[0], "max_us": s[-1],
            "q25_us": q[0], "q75_us": q[2], "iqr_us": q[2] - q[0],
            "ci95_lo_us": lo, "ci95_hi_us": hi, "ci_coverage": round(cover, 4),
            "repeats": len(s)}


def ratio_ci(num, den, seed=20260909, resamples=BOOTSTRAP):
    """Bootstrap 95% CI of median(den)/median(num) over paired repeats."""
    rng = random.Random(seed)
    n = min(len(num), len(den))
    pt = statistics.median(den) / statistics.median(num)
    boot = []
    idx = range(n)
    for _ in range(resamples):
        pick = [rng.randrange(n) for _ in idx]
        boot.append(statistics.median([den[i] for i in pick]) /
                    statistics.median([num[i] for i in pick]))
    boot.sort()
    lo = boot[int(0.025 * resamples)]
    hi = boot[int(0.975 * resamples)]
    return {"ratio": pt, "ci95_lo": lo, "ci95_hi": hi,
            "contains_1": lo <= 1.0 <= hi}


# ---------------------------------------------------------------------------
# the timer
# ---------------------------------------------------------------------------
def pin(cpu=CPU):
    import os
    try:
        os.sched_setaffinity(0, {cpu})
        return sorted(os.sched_getaffinity(0))
    except OSError as exc:                            # pragma: no cover
        return "unavailable: %s" % exc


def load():
    return pathlib.Path("/proc/loadavg").read_text().split()[:3]


def sweep_ns(call, recs):
    """One full sweep of the distribution, gc off, in nanoseconds."""
    gc.collect()
    gc.disable()
    try:
        t = time.perf_counter_ns()
        for r in recs:
            call(r)
        return time.perf_counter_ns() - t
    finally:
        gc.enable()


def measure(g, entry="call", repeats=REPEATS):
    """Per-record microseconds for every arm on every distribution."""
    dists = {"deploy": g["deploy"], "uniform": g["uniform"],
             "worst": [g["worst"]] * len(g["uniform"])}
    dists = {k: v for k, v in dists.items() if v}
    names = [a for a in ORDER if a in g["arms"] and g["arms"][a].get(entry)]

    # warm-up, discarded
    for d in dists.values():
        for a in names:
            g["arms"][a][entry](d[0])
            sweep_ns(g["arms"][a][entry], d[:64])

    samples = {d: {a: [] for a in names} for d in dists}
    for _ in range(repeats):
        for dname, recs in dists.items():
            for a in names:                            # round-robin inside repeat
                ns = sweep_ns(g["arms"][a][entry], recs)
                samples[dname][a].append(ns / 1e3 / len(recs))
    return {d: {a: summarize(v) for a, v in per.items()} for d, per in samples.items()}, samples


def bytecodes(g, recs, arm, entry="call", limit=512, seed=20260909):
    """Executed CPython bytecodes per record -- expected and worst over `recs`.

    Sampled with a seeded uniform draw, not a stride.  A stride over
    `itertools.product` order pins the low-order input digits and so cannot see
    the control-flow classes that make them true -- the same defect as
    AMENDMENT 1.  The seed is fixed so the draw is reproducible.
    """
    call = g["arms"][arm][entry]
    if len(recs) <= limit:
        picks = list(recs)
    else:
        picks = random.Random(seed).sample(list(recs), limit)
    counts = [lazy_cost.count_opcodes(lambda r=r: call(r)) for r in picks]
    return {"sampled": len(counts), "expected": statistics.mean(counts),
            "worst": max(counts), "best": min(counts)}


def run_stage(stage, repeats=REPEATS):
    t0 = time.time()
    print("== stage %d" % stage)
    gate, g = equivalence.check(stage)
    if not gate["gate_passed"]:
        return {"stage": stage, "gate": gate, "timed": False,
                "reason": "equivalence gate failed; no timing reported"}

    rep = {"stage": stage, "gate": gate, "timed": True,
           "cpu_affinity": pin(), "loadavg_before": load(),
           "repeats": repeats,
           "distribution_sizes": {"deploy": len(g["deploy"]) if g["deploy"] else 0,
                                  "uniform": len(g["uniform"]), "worst": 1},
           "deploy_what": g["deploy_what"], "uniform_what": g["uniform_what"],
           "worst_note": g["worst_note"]}

    print("   timing, envelope-matched (%d repeats)" % repeats)
    rep["latency_us"], raw = measure(g, "call", repeats)
    print("   timing, bare entry")
    rep["latency_bare_us"], rawb = measure(g, "bare", repeats)

    rep["ratios"] = {}
    for dname, per in raw.items():
        rep["ratios"][dname] = {}
        for a in per:
            if a == "A":
                continue
            rep["ratios"][dname]["A_over_" + a] = ratio_ci(per[a], per["A"])
            rep["ratios"][dname][a + "_over_A"] = ratio_ci(per["A"], per[a])

    print("   bytecodes")
    rep["bytecodes"] = {}
    for dname, recs in (("deploy", g["deploy"]), ("uniform", g["uniform"]),
                        ("worst", [g["worst"]])):
        if not recs:
            continue
        rep["bytecodes"][dname] = {a: bytecodes(g, recs, a) for a in g["arms"]
                                   if g["arms"][a].get("call")}

    rep["source_bytes"] = {a: len(v["source"]) for a, v in g["arms"].items()}
    rep["loadavg_after"] = load()
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


def main(stages=(1, 2, 3), repeats=REPEATS):
    pin()
    report = {"protocol": "research/lazy-latency/PREREGISTRATION.md",
              "python": sys.version.split()[0], "cpu_pinned_to": CPU,
              "nproc": __import__("os").cpu_count(),
              "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "loadavg_at_start": load()}
    for s in stages:
        report["stage%d" % s] = run_stage(s, repeats)
    report["loadavg_at_end"] = load()
    (OUT / "latency.json").write_text(json.dumps(report, indent=1, default=str))
    print("-> %s" % (OUT / "latency.json"))
    return report


if __name__ == "__main__":
    ss = tuple(int(x) for x in sys.argv[1:]) or (1, 2, 3)
    main(ss)
