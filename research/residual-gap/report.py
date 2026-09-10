"""Prints every number in RESULTS.md from `out/*.json`.  Re-runs nothing."""
from __future__ import annotations

import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
LADDER = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"]
LABEL = {
    "R1": "guard-call inlining (_ck/_ix/_dz/_dzi)",
    "R2": "module-call inlining (_m0)",
    "R3": "loop-invariant code motion + CSE",
    "R4": "short-circuit the truth-table conjunction",
    "R5": "typed-guard elimination (clamp retained)",
    "R6": "loop-bound strength reduction (while form)",
    "R7": "arm C verbatim -- unaccounted remainder",
}
CLS = {"R1": "repr", "R2": "repr", "R3": "repr", "R4": "STRUCT",
       "R5": "repr", "R6": "repr", "R7": "unacc"}


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def hr(t):
    print("\n" + t)
    print("-" * len(t))


def main():
    r1, r2 = load("latency_run1.json"), load("latency_run2.json")
    cnt, fit, op, gt, it = (load("counts.json"), load("fit.json"),
                            load("outproc.json"), load("gate.json"),
                            load("interp.json"))

    hr("0. Gate")
    print("ladder_gate_passed  :", gt["ladder_gate_passed"])
    for k, v in gt["rungs"].items():
        print("   %-4s %-4s mismatches %s %s" % (k, "PASS" if v["passed"] else "FAIL",
                                                 v["mismatches"],
                                                 v["error"] or ""))
    print("interpreter oracle  : %d records, %d strata, all_agree=%s"
          % (it["records"], len(it["strata"]), it["all_agree"]))

    hr("1. Reproduction of FINDINGS section 51 (envelope-matched, us/record)")
    print("%-10s %10s %10s %10s | %8s %8s %8s"
          % ("dist", "A", "B", "C", "A/B", "A/C", "B/C"))
    for tag, rep in (("run1", r1), ("run2", r2)):
        for d in ("deploy", "uniform", "worst"):
            s = rep["summary_us"][d]
            rr = rep["ratios"][d]
            print("%-10s %10.3f %10.3f %10.3f | %8.3f %8.3f %8.3f   (%s)"
                  % (d, s["env:A"]["median_us"], s["env:B"]["median_us"],
                     s["env:C"]["median_us"], rr["A_over_B_envelope"]["ratio"],
                     rr["A_over_C_envelope"]["ratio"],
                     rr["B_over_C_envelope"]["ratio"], tag))
    print("\nsection 51 recorded: 39.043 / 18.781 / 3.025, A/B 2.079, A/C 12.908, "
          "B/C 6.208")

    for d in ("deploy", "uniform", "worst"):
        hr("2. The ladder on the %s distribution (bare entry, us/record)" % d)
        s1 = r1["summary_us"][d]
        s2 = r2["summary_us"][d]
        tot = s1["bare:R0"]["median_us"] / s1["bare:R7"]["median_us"]
        L = math.log(tot)
        print("%-4s %-42s %9s %9s %9s %9s %8s %7s"
              % ("rung", "transformation", "run1 us", "run2 us", "ratio",
                 "CI lo-hi", "log %", "class"))
        acc = {"repr": 0.0, "STRUCT": 0.0, "unacc": 0.0}
        for i in range(len(LADDER) - 1):
            a, b = LADDER[i], LADDER[i + 1]
            key = "%s_over_%s" % (a, b)
            rat = r1["ratios"][d][key]
            share = math.log(rat["ratio"]) / L * 100
            acc[CLS[b]] += math.log(rat["ratio"])
            print("%-4s %-42s %9.3f %9.3f %9.3f %4.2f-%4.2f %7.1f %7s"
                  % (b, LABEL[b], s1["bare:" + b]["median_us"],
                     s2["bare:" + b]["median_us"], rat["ratio"],
                     rat["ci95_lo"], rat["ci95_hi"], share, CLS[b]))
        print("%-4s %-42s %9.3f %9.3f" % ("R0", "arm B verbatim (start)",
                                          s1["bare:R0"]["median_us"],
                                          s2["bare:R0"]["median_us"]))
        print("\nR0 / R7 = %.3f (run1)  %.3f (run2)   log-total %.4f"
              % (tot, s2["bare:R0"]["median_us"] / s2["bare:R7"]["median_us"], L))
        print("representational %.4f (%.1f%%)   structural %.4f (%.1f%%)   "
              "unaccounted %.4f (%.1f%%)"
              % (acc["repr"], 100 * acc["repr"] / L, acc["STRUCT"],
                 100 * acc["STRUCT"] / L, acc["unacc"], 100 * acc["unacc"] / L))
        print("strict reading (operation-count-changing rungs R3+R4 = structural): "
              "structural %.1f%%, representational %.1f%%"
              % (100 * (math.log(r1["ratios"][d]["R2_over_R3"]["ratio"])
                        + math.log(r1["ratios"][d]["R3_over_R4"]["ratio"])) / L,
                 100 * (acc["repr"] + acc["STRUCT"]
                        - math.log(r1["ratios"][d]["R2_over_R3"]["ratio"])
                        - math.log(r1["ratios"][d]["R3_over_R4"]["ratio"])) / L))

    hr("3. Boundary versus body -- linear fit t(k) = a + b k")
    print("%-4s %12s %14s %20s" % ("rung", "intercept us", "slope us/iter",
                                   "body share at k=%.1f" % fit["deploy_mean_k"]))
    for n in LADDER + ["A"]:
        f = fit["fits"][n]
        print("%-4s %12.3f %14.4f %19.1f%%"
              % (n, f["intercept_us"], f["slope_us_per_iteration"],
                 100 * f["body_fraction_at_deploy"]))
    fr0, fr7 = fit["fits"]["R0"], fit["fits"]["R7"]
    print("intercept ratio R0/R7 = %.2f ; slope ratio R0/R7 = %.2f ; "
          "overall time ratio = %.2f"
          % (fr0["intercept_us"] / fr7["intercept_us"],
             fr0["slope_us_per_iteration"] / fr7["slope_us_per_iteration"],
             r1["ratios"]["deploy"]["R0_over_R7_bare"]["ratio"]))

    hr("4. Three cost measures, deployment distribution")
    p = cnt["profiles"]["deploy"]
    pw = cnt["profiles"]["worst"]
    kd = cnt["loop_iterations"]["deploy"]["mean"]
    kw = cnt["loop_iterations"]["worst"]
    print("%-4s %11s %10s %10s %9s %11s %10s"
          % ("rung", "bytecodes", "bc/step", "py calls", "time us",
             "ns/bytecode", "peak B"))
    for n in LADDER:
        bd = p["bare:" + n]["bytecodes_per_record"]
        bw = pw["bare:" + n]["bytecodes_per_record"]
        slope = (bw - bd) / (kw - kd)
        t = r1["summary_us"]["deploy"]["bare:" + n]["median_us"]
        print("%-4s %11.1f %10.1f %10.2f %9.3f %11.2f %10.0f"
              % (n, bd, slope, p["bare:" + n]["py_calls_per_record"], t,
                 t * 1000 / bd, cnt["alloc"]["bare:" + n]["median_bytes"]))
    for n in ("env:A", "env:B", "env:C"):
        bd = p[n]["bytecodes_per_record"]
        t = r1["summary_us"]["deploy"][n]["median_us"]
        print("%-4s %11.1f %10s %10.2f %9.3f %11.2f %10.0f"
              % (n, bd, "-", p[n]["py_calls_per_record"], t, t * 1000 / bd,
                 cnt["alloc"][n]["median_bytes"]))

    hr("5. Profile diff -- executed bytecodes per record by code object")
    for n in ("bare:R0", "bare:R7", "env:A"):
        d = p[n]["by_code_per_record"]
        tot = p[n]["bytecodes_per_record"]
        keep = {k: v for k, v in d.items()
                if not k.startswith("profile_bytecodes")
                and "<lambda>" not in k}
        print("%-8s total %8.1f : %s" % (n, tot, ", ".join(
            "%s %.1f (%.1f%%)" % (k, v, 100 * v / tot) for k, v in keep.items())))
    g0 = p["bare:R0"]["by_code_per_record"]
    helper = sum(v for k, v in g0.items() if k in ("_ck", "_ix", "_dz", "_dzi"))
    print("\nguard-helper frames in arm B: %.1f bytecodes/record = %.1f%% of its "
          "%.1f, and %.1f Python calls/record against arm C's %.1f"
          % (helper, 100 * helper / p["bare:R0"]["bytecodes_per_record"],
             p["bare:R0"]["bytecodes_per_record"],
             p["bare:R0"]["py_calls_per_record"],
             p["bare:R7"]["py_calls_per_record"]))
    for n in ("bare:R0", "bare:R7"):
        ops = p[n]["by_op_per_record"]
        print("%-8s BINARY_SUBSCR %.1f  BUILD_TUPLE %.1f  CALL %.1f  COMPARE_OP %.1f"
              % (n, ops.get("BINARY_SUBSCR", 0), ops.get("BUILD_TUPLE", 0),
                 ops.get("CALL", 0), ops.get("COMPARE_OP", 0)))

    hr("5b. Per-scan-step opcode slopes (deploy -> worst, k=%.2f -> %d)" % (kd, kw))
    watch = ("BINARY_SUBSCR", "BUILD_TUPLE", "CALL", "COMPARE_OP", "LOAD_FAST",
             "FOR_ITER", "JUMP_BACKWARD", "POP_JUMP_IF_FALSE", "BINARY_OP")
    print("%-18s %12s %12s %12s %12s" % ("opcode", "R0 /record", "R0 /step",
                                         "R7 /record", "R7 /step"))
    for o in watch:
        a0 = p["bare:R0"]["by_op_per_record"].get(o, 0.0)
        b0 = pw["bare:R0"]["by_op_per_record"].get(o, 0.0)
        a7 = p["bare:R7"]["by_op_per_record"].get(o, 0.0)
        b7 = pw["bare:R7"]["by_op_per_record"].get(o, 0.0)
        print("%-18s %12.1f %12.2f %12.1f %12.2f"
              % (o, a0, (b0 - a0) / (kw - kd), a7, (b7 - a7) / (kw - kd)))

    hr("6. Out-of-process replication (%s, no venv, no tcn)" % op["python"])
    print("leaked modules:", op["leaked"])
    L2 = math.log(op["R0_over_R7"])
    for i in range(len(LADDER) - 1):
        a, b = LADDER[i], LADDER[i + 1]
        rr = op["ratios"]["%s_over_%s" % (a, b)]
        print("   %-4s %8.3f us   ratio %6.3f   log %5.1f%%   %s"
              % (b, op["median_us"][b], rr, 100 * math.log(rr) / L2, CLS[b]))
    print("   R0 %8.3f us ; R0/R7 = %.3f (in process %.3f)"
          % (op["median_us"]["R0"], op["R0_over_R7"],
             r1["ratios"]["deploy"]["R0_over_R7_bare"]["ratio"]))

    hr("7. Loop iterations -- candidate 3")
    print(json.dumps(cnt["loop_iterations"], indent=1))


if __name__ == "__main__":
    main()
