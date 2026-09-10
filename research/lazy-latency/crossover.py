"""PREREGISTRATION section 6 -- at what sparsity does laziness stop paying?

Two sweeps, both fixed in advance:

* **stage 3** -- bin the 2,883 held-out records by achieved extent
  `max(w, h)`, time A, B and C per bin, and report the smallest extent `k*` at
  which B's median per-record latency exceeds A's.  Arm A is a fixed 31-term
  formulation (constant work) and arm B is linear in the extent, so a crossover
  must exist inside [1, 31] or B wins at every extent.  Then place the
  deployment distribution against `k*` using its own measured extent histogram.

* **stage 1** -- sweep the predicate hit rate and report `p*`, the rate at
  which B's median crosses A's, by linear interpolation between the bracketing
  points.

Same timer, same pinning, same gc discipline as `latency.py`.
"""
from __future__ import annotations

import collections
import json
import pathlib
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arms                                  # noqa: E402
import latency                               # noqa: E402

OUT = arms.OUT
REPEATS = 15                                 # per bin; bins are many and small
MIN_BIN = 1                                  # AMENDMENT 3: every bin is timed, padded to CALLS
HIT_RATES = [0.0, 1.0 / 48.0, 1.0 / 16.0, 0.125, 0.25, 0.375, 0.5,
             0.625, 0.75, 0.875, 1.0]


CALLS = 2048    # AMENDMENT 3: every bin is padded to this many calls


def pad(recs, calls=CALLS):
    """Cycle a bin's records up to `calls` calls.

    AMENDMENT 3, see RESULTS.md.  The pre-registration dropped bins smaller than
    `MIN_BIN`, and the stage-3 crossover turns out to lie *inside* those bins:
    every `w + h` above 32 has one or two records.  Padding by cycling gives
    every bin the same sweep length and the same instrument, and it is exactly
    what the pre-registered `D_worst` arm already does (one record called
    `len(D_uniform)` times).  It is applied identically to all three arms.
    """
    if len(recs) >= calls:
        return list(recs)          # never truncate; padding only ever lengthens
    out = list(recs)
    while len(out) < calls:
        out.extend(recs)
    return out[:calls]


def time_arms(g, recs, names, repeats=REPEATS, entry="call"):
    recs = pad(recs)
    for a in names:
        g["arms"][a][entry](recs[0])
        latency.sweep_ns(g["arms"][a][entry], recs[:32])
    s = {a: [] for a in names}
    for _ in range(repeats):
        for a in names:
            s[a].append(latency.sweep_ns(g["arms"][a][entry], recs) / 1e3 / len(recs))
    return s


def ols(xs, ys):
    """Least-squares intercept and slope; arm B's cost is linear in iterations."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    return my - b * mx, b


def interpolate(xs, ya, yb):
    """First x where yb crosses above ya, linearly interpolated."""
    for i in range(1, len(xs)):
        d0 = yb[i - 1] - ya[i - 1]
        d1 = yb[i] - ya[i]
        if d0 <= 0 < d1:
            return xs[i - 1] + (xs[i] - xs[i - 1]) * (-d0) / (d1 - d0)
    return None


# ---------------------------------------------------------------------------
def stage3_crossover(repeats=REPEATS, key=None, label="w+h", min_bin=None, g=None):
    key = key or arms.stage3_iterations
    min_bin = MIN_BIN if min_bin is None else min_bin
    print("== stage 3 crossover, binned by %s" % label)
    g = g or arms.build(3)
    names = [a for a in ("A", "B", "C") if a in g["arms"]]
    bins = collections.defaultdict(list)
    for r in g["uniform"]:
        bins[key(r)].append(r)
    dep = collections.Counter(key(r) for r in g["deploy"])

    rows = []
    for k in sorted(bins):
        recs = bins[k]
        row = {"extent": k, "records": len(recs),
               "deploy_records": dep.get(k, 0), "timed": len(recs) >= min_bin}
        if row["timed"]:
            s = time_arms(g, recs, names, repeats)
            for a in names:
                row[a + "_us"] = statistics.median(s[a])
            row["A_over_B"] = row["A_us"] / row["B_us"]
            row["A_over_C"] = row["A_us"] / row["C_us"]
        rows.append(row)
        print("   extent %2d  n=%4d  %s" % (
            k, len(recs),
            "  ".join("%s %7.3f us" % (a, row[a + "_us"]) for a in names)
            if row["timed"] else "(bin too small to time)"))

    timed = [r for r in rows if r["timed"]]
    xs = [r["extent"] for r in timed]
    star = interpolate(xs, [r["A_us"] for r in timed], [r["B_us"] for r in timed])
    star_c = interpolate(xs, [r["A_us"] for r in timed], [r["C_us"] for r in timed])
    above = [r for r in timed if r["A_over_B"] < 1.0]
    fit = ols(xs, [r["B_us"] for r in timed])
    fit_c = ols(xs, [r["C_us"] for r in timed])
    a_mean = statistics.mean(r["A_us"] for r in timed)
    fitted = {"B_intercept_us": fit[0], "B_slope_us_per_iteration": fit[1],
              "C_intercept_us": fit_c[0], "C_slope_us_per_iteration": fit_c[1],
              "A_mean_us": a_mean,
              "crossover_from_fit": (a_mean - fit[0]) / fit[1] if fit[1] else None,
              "crossover_C_from_fit": (a_mean - fit_c[0]) / fit_c[1] if fit_c[1] else None}

    # where the deployment distribution sits
    dep_ext = [key(r) for r in g["deploy"]]
    uni_ext = [key(r) for r in g["uniform"]]
    return {"binned_by": label, "rows": rows, "min_bin": min_bin, "repeats": repeats,
            "crossover_extent_B": star,
            "crossover_extent_C": star_c,
            "first_extent_where_B_slower": above[0]["extent"] if above else None,
            "deploy_extent_histogram": dict(sorted(collections.Counter(dep_ext).items())),
            "uniform_extent_histogram": dict(sorted(collections.Counter(uni_ext).items())),
            "deploy_extent_mean": statistics.mean(dep_ext),
            "uniform_extent_mean": statistics.mean(uni_ext),
            "deploy_fraction_at_or_above_crossover":
                (sum(1 for e in dep_ext if star is not None and e >= star) / len(dep_ext))
                if star is not None else 0.0,
            "linear_fit": fitted,
            "note": "arm A is a fixed 31-term formulation, so its cost does not "
                    "depend on the extent; arm B is linear in it"}


# ---------------------------------------------------------------------------
def stage1_crossover(repeats=REPEATS):
    print("== stage 1 crossover, by predicate hit rate")
    g = arms.build(1)
    names = ["A", "B"]
    fx = g["fixture"]
    rows = []
    for p in HIT_RATES:
        recs = arms.hit_rate_inputs(fx, p, arms.SUBSAMPLE)
        hits = sum(1 for r in recs for v in r if v == g["hit"]) / (len(recs) * g["n"])
        s = time_arms(g, recs, names, repeats)
        row = {"requested_hit_rate": p, "achieved_hit_rate": hits,
               "records": len(recs),
               "A_us": statistics.median(s["A"]), "B_us": statistics.median(s["B"])}
        row["A_over_B"] = row["A_us"] / row["B_us"]
        rows.append(row)
        print("   p=%.4f (achieved %.4f)  A %7.3f us  B %7.3f us  A/B %.3f"
              % (p, hits, row["A_us"], row["B_us"], row["A_over_B"]))
    xs = [r["achieved_hit_rate"] for r in rows]
    star = interpolate(xs, [r["A_us"] for r in rows], [r["B_us"] for r in rows])
    return {"rows": rows, "repeats": repeats, "crossover_hit_rate": star,
            "deployment_sparsity_reference": {
                "parse_corner_rate": 20 / 961,
                "lazy_guard_corner_rate": 54 / 2883,
                "miniature_uniform_rate": 1 / 16}}


def main():
    latency.pin()
    t0 = time.time()
    rep = {"protocol": "research/lazy-latency/PREREGISTRATION.md section 6",
           "loadavg_at_start": latency.load(), "cpu_pinned_to": latency.CPU}
    g3 = arms.build(3)
    rep["stage3"] = stage3_crossover(g=g3)
    rep["stage3_by_max_extent"] = stage3_crossover(
        key=arms.stage3_extent, label="max(w,h)", g=g3)
    rep["stage1"] = stage1_crossover()
    rep["loadavg_at_end"] = latency.load()
    rep["seconds"] = round(time.time() - t0, 1)
    (OUT / "crossover.json").write_text(json.dumps(rep, indent=1, default=str))
    print("-> %s" % (OUT / "crossover.json"))
    return rep


if __name__ == "__main__":
    main()
