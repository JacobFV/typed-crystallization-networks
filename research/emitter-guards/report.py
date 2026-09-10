"""Print every number in RESULTS.md from `out/*.json`, so it is checkable
without re-running anything."""
from __future__ import annotations

import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "out"
ART = ("mixed", "language", "computer", "visual")


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    g = load("gate.json")
    print("== gates (out/gate.json), baseline %s sha %s" % (g["baseline_rev"], g["baseline_source_sha256"][:12]))
    print("%-9s %8s %12s %8s %10s %8s" % ("artifact", "cases", "compares", "A-mism", "interp", "B-mism"))
    for a in ART:
        v = g["artifacts"][a]
        print("%-9s %8d %12d %8d %10d %8d   %s / %s" % (
            a, v["cases"], v["comparisons"], v["gate_A_mismatches"], v["gate_B_checked"],
            v["gate_B_mismatches"],
            "A PASS" if v["gate_A_passed"] else "A FAIL",
            "B PASS" if v["gate_B_passed"] else "B FAIL"))
    print("all_passed: %s" % g["all_passed"])

    print("\n== guards, per class (emitted by the baseline = emitted + eliminated)")
    print("%-9s %28s %28s" % ("artifact", "G1 range  elim / total", "G2 index  elim / total"))
    for a in ART:
        s = g["artifacts"][a]["guards_new"]
        g1e, g1k = s["guard_range_eliminated"], s["guard_range_emitted"]
        g2e, g2k = s["guard_index_eliminated"], s["guard_index_emitted"]
        print("%-9s %28s %28s" % (a, "%d / %d" % (g1e, g1e + g1k), "%d / %d" % (g2e, g2e + g2k)))

    print("\n== three cost measures (out/measure_run*.json)")
    runs = [(t, load("measure_%s.json" % t)) for t in ("run1", "run2", "run3")]
    runs = [(t, r) for t, r in runs if r]
    for a in ART:
        base = runs[0][1]["artifacts"][a]
        pr = base["primitives"]
        print("\n-- %-8s  %s" % (a, "NULL CONTROL: both arms emit byte-identical source"
                                 if base["null_control"] else "source differs between arms"))
        print("   primitives  worst %d  expected %.1f   (program-level; arm-independent)"
              % (pr["worst_ops"], pr["expected_ops"]))
        b = base["bytecodes"]
        print("   bytecodes   old %.1f -> new %.1f   x%.4f" %
              (b["old"]["expected"], b["new"]["expected"], b["ratio_old_over_new"]))
        for t, r in runs:
            v = r["artifacts"][a]
            q = v["latency_ratio_old_over_new"]
            print("   wall %-4s   old %10.4f us -> new %10.4f us   x%.4f CI[%.4f, %.4f]%s"
                  % (t, v["latency_us"]["old"]["median_us"], v["latency_us"]["new"]["median_us"],
                     q["ratio"], q["ci95_lo"], q["ci95_hi"],
                     "  (CI contains 1)" if q["contains_1"] else ""))
        d = base["deployment"]
        print("   disk        source %d -> %d B   pyz %d -> %d B   cold %.1f -> %.1f ms"
              % (d["old"]["source_bytes"], d["new"]["source_bytes"],
                 d["old"]["pyz_bytes"], d["new"]["pyz_bytes"],
                 d["old"]["cold_start_ms"], d["new"]["cold_start_ms"]))

    at = load("attribute.json")
    if at:
        print("\n== where the guards are, against where the time is (out/attribute.json)")
        for a, v in at.items():
            print("-- %s  calls/case %s" % (a, json.dumps(v["calls_per_case"]["new"])))
            print("   eliminated %s" % json.dumps(v["guards_eliminated_by_function"]))
            print("   kept       %s" % json.dumps(v["guards_kept_by_function"]))
            print("   bytecodes old %s" % json.dumps(v["bytecodes_by_code_object"]["old"]))
            print("   bytecodes new %s" % json.dumps(v["bytecodes_by_code_object"]["new"]))
            for fn, s in v.get("subroutine_timing", {}).items():
                if s.get("identical"):
                    print("   subroutine %s  %.4f -> %.4f us  x%.4f CI[%.4f, %.4f]  bytecodes/call %.1f -> %.1f"
                          % (fn, s["latency_us"]["old"]["median_us"], s["latency_us"]["new"]["median_us"],
                             s["ratio_old_over_new"]["ratio"], s["ratio_old_over_new"]["ci95_lo"],
                             s["ratio_old_over_new"]["ci95_hi"], s["bytecodes_per_call"]["old"],
                             s["bytecodes_per_call"]["new"]))

    res = load("residue.json")
    if res:
        print("\n== guards the declared type does not bound (out/residue.json)")
        for a in ART:
            v = res.get(a)
            if not v or not v["kept_by_shape"]:
                continue
            print("-- %s" % a)
            for k, n in v["kept_by_shape"].items():
                print("   %5d  %s" % (n, k))


if __name__ == "__main__":
    main()
