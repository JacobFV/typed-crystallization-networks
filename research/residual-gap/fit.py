"""Candidate 2 -- how much of each rung is boundary rather than body?

Every rung is a straight-line prologue and epilogue around two scans whose
length is `w + h`.  Timing each rung against `k = w + h` and fitting
`t(k) = intercept + slope * k` separates the two directly: the **intercept** is
the non-loop cost (the typed prologue, the packing epilogue and the call
itself) and the **slope** is the per-scan-step cost.  That is the boundary /
body split candidate 2 asks for, measured rather than asserted.

`research/lazy-latency/crossover.py` supplies `pad` (AMENDMENT 3's bin padding)
and `ols`; the timer and gc discipline are `latency.sweep_ns` as everywhere
else in this ladder.
"""
from __future__ import annotations

import collections
import json
import pathlib
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "research" / "lazy-latency"),
           str(ROOT / "research" / "compiled-runtime")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms                                                  # noqa: E402
import crossover as CX                                       # noqa: E402
import latency as LL                                         # noqa: E402
import gate as GATE                                          # noqa: E402
import rungs as RUNGS                                        # noqa: E402

OUT = HERE / "out"
CALLS = 1024
REPEATS = 11
TARGET_K = [2, 4, 6, 8, 12, 16, 20, 24, 30, 36, 42, 47]


def main():
    grep, g, R = GATE.check()
    if not grep["ladder_gate_passed"]:
        raise SystemExit("gate failed")

    by_k = collections.defaultdict(list)
    for r in g["uniform"]:
        by_k[arms.stage3_iterations(r)].append(r)
    ks = [k for k in TARGET_K if k in by_k]

    entries = {n: R[n]["module"].run for n in RUNGS.LADDER}
    entries["A"] = g["arms"]["A"]["call"]

    rows = {n: {} for n in entries}
    for k in ks:
        recs = CX.pad(by_k[k], CALLS)
        for n, call in entries.items():
            call(recs[0])
            LL.sweep_ns(call, recs[:32])
        s = {n: [] for n in entries}
        for _ in range(REPEATS):
            for n, call in entries.items():
                s[n].append(LL.sweep_ns(call, recs) / 1e3 / len(recs))
        for n in entries:
            rows[n][k] = statistics.median(s[n])
        print("   k=%2d  n=%3d  %s" % (k, len(by_k[k]),
                                       "  ".join("%s=%.2f" % (n, rows[n][k])
                                                 for n in RUNGS.LADDER)), flush=True)

    fits = {}
    for n in entries:
        xs = [float(k) for k in ks]
        ys = [rows[n][k] for k in ks]
        a, b = CX.ols(xs, ys)
        fits[n] = {"intercept_us": a, "slope_us_per_iteration": b,
                   "bins": {str(k): rows[n][k] for k in ks}}

    it_d = [arms.stage3_iterations(r) for r in g["deploy"]]
    kbar = statistics.mean(it_d)
    for n in fits:
        a, b = fits[n]["intercept_us"], fits[n]["slope_us_per_iteration"]
        fits[n]["predicted_at_deploy_mean_k"] = a + b * kbar
        fits[n]["body_fraction_at_deploy"] = (b * kbar) / (a + b * kbar)

    rep = {"protocol": "research/residual-gap/PREREGISTRATION.md section 6 candidate 2",
           "calls_per_bin": CALLS, "repeats": REPEATS, "bins": ks,
           "bin_sizes": {str(k): len(by_k[k]) for k in ks},
           "deploy_mean_k": kbar,
           "cpu_affinity": LL.pin(), "loadavg": LL.load(),
           "fits": fits}
    (OUT / "fit.json").write_text(json.dumps(rep, indent=1, default=str))
    print("->", OUT / "fit.json")
    print("\nrung  intercept us   slope us/iter   body fraction at k=%.1f" % kbar)
    for n in RUNGS.LADDER + ["A"]:
        f = fits[n]
        print("  %-3s %10.3f %14.4f %14.1f%%"
              % (n, f["intercept_us"], f["slope_us_per_iteration"],
                 100 * f["body_fraction_at_deploy"]))
    return rep


if __name__ == "__main__":
    main()
