"""Print every number quoted in RESULTS.md from `out/*.json`.

Nothing is recomputed here.  A reader can check the tables against the raw JSON
without re-running any arm.
"""
from __future__ import annotations

import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "out"


def load(name):
    p = OUT / ("%s.json" % name)
    return json.loads(p.read_text()) if p.exists() else None


def latency():
    d = load("latency")
    if not d:
        return
    print("### latency  (python %s, pinned to cpu %s, load %s -> %s)"
          % (d["python"], d["cpu_pinned_to"], d["loadavg_at_start"], d["loadavg_at_end"]))
    for s in (1, 2, 3):
        k = "stage%d" % s
        if k not in d:
            continue
        r = d[k]
        print("\n-- stage %d  (%d repeats; deploy=%s uniform=%s)"
              % (s, r["repeats"], r["distribution_sizes"]["deploy"],
                 r["distribution_sizes"]["uniform"]))
        print("   worst case: %s = %s" % (r["worst_note"]["criterion"],
                                          r["worst_note"]["value"]))
        for dist in ("deploy", "uniform", "worst"):
            if dist not in r["latency_us"]:
                continue
            print("   [%s]" % dist)
            for a, v in sorted(r["latency_us"][dist].items()):
                bare = r["latency_bare_us"].get(dist, {}).get(a)
                print("     %s  median %9.4f us  min %9.4f  IQR %8.4f  "
                      "CI95 [%9.4f, %9.4f]%s"
                      % (a, v["median_us"], v["min_us"], v["iqr_us"],
                         v["ci95_lo_us"], v["ci95_hi_us"],
                         "  bare %9.4f" % bare["median_us"] if bare else ""))
            for a in ("B", "C"):
                key = "A_over_%s" % a
                if key in r["ratios"].get(dist, {}):
                    x = r["ratios"][dist][key]
                    print("     A / %s = %6.3f x   CI95 [%.3f, %.3f]   contains 1.0: %s"
                          % (a, x["ratio"], x["ci95_lo"], x["ci95_hi"], x["contains_1"]))
            bc = r["bytecodes"].get(dist, {})
            if bc:
                print("     bytecodes/record  " + "  ".join(
                    "%s exp %.0f worst %d" % (a, bc[a]["expected"], bc[a]["worst"])
                    for a in sorted(bc)))


def crossover():
    d = load("crossover")
    if not d:
        return
    print("\n### crossover")
    s3 = d["stage3"]
    print("-- stage 3, by achieved extent max(w,h); arm A is constant-work")
    print("   %-7s %6s %7s %10s %10s %10s %8s" %
          ("extent", "n", "deploy", "A us", "B us", "C us", "A/B"))
    for r in s3["rows"]:
        if r["timed"]:
            print("   %-7d %6d %7d %10.4f %10.4f %10.4f %8.3f"
                  % (r["extent"], r["records"], r["deploy_records"],
                     r["A_us"], r["B_us"], r["C_us"], r["A_over_B"]))
        else:
            print("   %-7d %6d %7d   (bin below %d, not timed)"
                  % (r["extent"], r["records"], r["deploy_records"], s3["min_bin"]))
    print("   crossover extent k* (B vs A): %s" % s3["crossover_extent_B"])
    print("   first timed extent where B is slower: %s" % s3["first_extent_where_B_slower"])
    print("   crossover extent for the hand-written reference: %s" % s3["crossover_extent_C"])
    print("   mean extent: deploy %.2f   uniform %.2f"
          % (s3["deploy_extent_mean"], s3["uniform_extent_mean"]))
    print("   deploy fraction at or above k*: %.4f"
          % s3["deploy_fraction_at_or_above_crossover"])
    s1 = d["stage1"]
    print("-- stage 1, by predicate hit rate")
    print("   %-10s %-10s %10s %10s %8s" % ("requested", "achieved", "A us", "B us", "A/B"))
    for r in s1["rows"]:
        print("   %-10.4f %-10.4f %10.4f %10.4f %8.3f"
              % (r["requested_hit_rate"], r["achieved_hit_rate"], r["A_us"],
                 r["B_us"], r["A_over_B"]))
    print("   crossover hit rate p*: %s" % s1["crossover_hit_rate"])
    print("   deployment sparsity references: %s" % s1["deployment_sparsity_reference"])


def deployment():
    d = load("deployment")
    if not d:
        return
    print("\n### deployment")
    for s in (1, 2, 3):
        k = "stage%d" % s
        if k not in d:
            continue
        print("-- stage %d" % s)
        for a, v in sorted(d[k].items()):
            if not isinstance(v, dict):
                continue
            print("   %s  %8d B  cold %7.3f ms (import %6.3f + first call %6.3f)  "
                  "peak RSS %7.2f MB  sweep %8.2f ms  leaked %s"
                  % (a, v["bytes_on_disk"], v["cold_start_ms_median"],
                     v["import_ms_median"], v["first_call_ms_median"],
                     v["peak_rss_mb_median"], v["sweep_ms_median"],
                     v["leaked_modules"]))


def equivalence():
    d = load("equivalence")
    if not d:
        return
    print("\n### equivalence gate  (all passed: %s)" % d["all_gates_passed"])
    for s in (1, 2, 3):
        k = "stage%d" % s
        r = d[k]
        print("-- stage %d  interpreter == A on %d: %s"
              % (s, r["interpreter_equals_A"]["checked"],
                 r["interpreter_equals_A"]["exact"]))
        for dn, v in sorted(r["arm_equivalence"].items()):
            print("     %-12s %6d records  bit-identical %s  digest %s"
                  % (dn, v["records"], v["bit_identical"],
                     sorted(v["digests"].values())[0][:16]))


if __name__ == "__main__":
    equivalence()
    latency()
    crossover()
    deployment()
