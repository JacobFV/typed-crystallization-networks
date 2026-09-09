"""Render the measured JSON in `out/` as the tables used in RESULTS.md."""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def group(rows):
    g = {}
    for r in rows:
        g.setdefault(r["method"], []).append(r)
    return g


def head(title):
    print("\n" + title)
    print("-" * len(title))


def summarise(rows, extra=()):
    print(f"| {'method':<34} | {'n':>2} | {'solved':>7} | {'median s':>10} | {'p90 s':>10} | "
          f"{'programs':>10} | {'op applications':>16} |")
    print(f"|{'-'*36}|{'-'*4}|{'-'*9}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*18}|")
    for m, rs in group(rows).items():
        secs = sorted(r["seconds"] for r in rs)
        p90 = secs[min(len(secs) - 1, int(0.9 * len(secs)))]
        print(f"| {m:<34} | {len(rs):>2} | {sum(r['solved'] for r in rs):>3}/{len(rs):<3} | "
              f"{st.median(secs):>10.5f} | {p90:>10.5f} | "
              f"{st.median(r['budget']['programs'] for r in rs):>10.0f} | "
              f"{st.median(r['budget']['op_applications'] for r in rs):>16.0f} |")


def main():
    j = load("joint.json")
    if j:
        head("JOINT (examples/joint.py) -- 16 x 16 = 256 discrete assignments")
        summarise(j["results"])
        print("\nconstants at declared initialization already implement the correct decision rule:",
              j["constants_probe"]["policy_correct_at_initialization"])
        print("\nselection-episode sweep (how many of 256 look optimal at E episodes):")
        for r in j["selection_noise_sweep"]:
            print(f"  E={r['selection_episodes']:>3}  perfect={r['perfect_scoring']:>3}  "
                  f"reference(6,6) among them={r['reference_included']}  "
                  f"first perfect={r['first_perfect']}  full sweep {r['seconds_full_sweep']:.1f}s")
        for r in j["results"]:
            if r["method"] == "exhaustive_enumeration_env_full":
                print("\n  behaviourally optimal assignments found by the full env sweep:",
                      r["detail"]["perfect_examples"], "of", r["detail"]["space_size"])

    m = load("mixed.json")
    if m:
        head("MIXED (examples/mixed.py) -- 16 x 1 x 3 x 2 = 96 discrete assignments")
        summarise(m["results"])

    mc = load("mixed_constants.json")
    if mc:
        head("MIXED + TRAINABLE CONSTANT -- 96 structures x 1 continuous parameter")
        summarise(mc["results"])
        for r in mc["results"]:
            if r["method"].startswith("hybrid"):
                print("  hybrid solution:", r["detail"]["solution"], "k error",
                      f"{r['detail']['k_error']:.2e}")
                break
        errs = [r["detail"]["k_error"] for r in mc["results"] if r["method"] == "gradient_tcn" and r["detail"]["k_error"] is not None]
        if errs:
            print(f"  gradient k error: median {st.median(errs):.2e}  max {max(errs):.2e}")

    for source in ("generator", "hard", "uniform"):
        for name in OUT.glob(f"scaling_{source}_*.json"):
            d = json.loads(name.read_text())
            head(f"SCALING -- targets from: {d['source']}   ({name.name})")
            print(f"| {'d':>2} | {'log2 space':>10} | {'density':>9} | {'enum s':>10} | {'enum leaves':>12} | "
                  f"{'sat s':>9} | {'sat confl':>9} | {'rand hit':>8} | {'grad succ':>9} | {'grad s':>8} |")
            print("|" + "|".join("-" * w for w in (4, 12, 11, 12, 14, 11, 11, 10, 11, 10)) + "|")
            bydepth = {}
            for r in d["rows"]:
                bydepth.setdefault(r["depth"], []).append(r)
            for depth in sorted(bydepth):
                rs = bydepth[depth]
                print(f"| {depth:>2} | {rs[0]['log2_space']:>10.1f} | "
                      f"{st.median(r['solution_density'] for r in rs):>9.1e} | "
                      f"{st.median(r['enumeration']['seconds'] for r in rs):>10.4f} | "
                      f"{st.median(r['enumeration']['leaf_programs'] for r in rs):>12.0f} | "
                      f"{st.median(r['sat']['seconds'] for r in rs):>9.4f} | "
                      f"{st.median(r['sat']['conflicts'] for r in rs):>9.0f} | "
                      f"{sum(r['random']['solved'] for r in rs)}/{len(rs):<6} | "
                      f"{st.mean(r['gradient']['success_rate'] for r in rs) if rs[0]['gradient'].get('success_rate') is not None else float('nan'):>9.2f} | "
                      f"{st.median(r['gradient']['seconds_median'] for r in rs) if rs[0]['gradient'].get('seconds_median') is not None else float('nan'):>8.2f} |")
            timeouts = [(r["depth"], r["target_index"]) for r in d["rows"] if r["enumeration"]["timed_out"]]
            if timeouts:
                print("  enumeration timeouts (budget exhausted):", timeouts)
            # enumeration throughput and projected full-sweep cost
            rates = [r["enumeration"]["leaf_programs"] / max(1e-9, r["enumeration"]["seconds"])
                     for r in d["rows"] if r["enumeration"]["leaf_programs"] > 1000]
            if rates:
                rate = st.median(rates)
                print(f"\n  measured enumeration throughput: {rate:,.0f} leaf programs/s")
                print(f"  {'d':>2} | {'space':>12} | projected FULL sweep")
                for depth in range(1, 9):
                    size = math.prod(16 * (4 + k) ** 2 for k in range(depth))
                    secs = size / rate
                    unit = (f"{secs:.4f} s" if secs < 60 else
                            f"{secs/60:.1f} min" if secs < 3600 else
                            f"{secs/3600:.1f} h" if secs < 86400 else
                            f"{secs/86400:.1f} days" if secs < 3.15e7 else
                            f"{secs/3.15e7:.2e} years")
                    print(f"  {depth:>2} | {size:>12.3e} | {unit}")

    n = load("noise_d2.json") or load("noise_d3.json")
    if n:
        head("NOISY TARGETS -- can each method recover the uncorrupted function?")
        byf = {}
        for r in n["rows"]:
            byf.setdefault(r["flips"], []).append(r)
        print(f"| {'flipped rows':>12} | {'enum recovers':>13} | {'sat SAT?':>9} | {'sat recovers':>12} | {'grad recovers':>13} |")
        print("|" + "|".join("-" * w for w in (14, 15, 11, 14, 15)) + "|")
        for f in sorted(byf):
            rs = byf[f]
            print(f"| {f:>12} | {sum(r['enumeration']['recovered_clean'] for r in rs)}/{len(rs):<11} | "
                  f"{sum(r['sat_exact']['solved'] for r in rs)}/{len(rs):<7} | "
                  f"{sum(r['sat_exact']['recovered_clean'] for r in rs)}/{len(rs):<10} | "
                  f"{st.mean(r['gradient']['recovered_clean_rate'] for r in rs):>13.2f} |")


if __name__ == "__main__":
    main()
