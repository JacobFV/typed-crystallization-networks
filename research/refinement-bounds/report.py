"""Print every number in RESULTS.md from `out/*.json`, so none of them is typed.

    .venv/bin/python research/refinement-bounds/report.py
"""
from __future__ import annotations

import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "out"


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def rule(t):
    print("\n" + t)
    print("-" * len(t))


def main():
    # ---------------------------------------------------------------- step 1
    s = load("soundness.json")
    st = load("static_ranges.json")
    if s:
        rule("1. SOUNDNESS -- what the min(., 3069) clamp guarantees")
        ci = s["clamp_identity"]
        print("clamp identity min(p+d, L) == p + min(d, L-p), mismatches / pairs:")
        for k in ("reachable_interior_961", "reachable_all_1024", "full_rectangle_0_to_L"):
            v = ci[k]
            print("   %-24s %d / %-10d  max pre-clamp %5d  max reformulated %5d"
                  % (k, v["mismatches"], v["pairs"], v["max_preclamp_value"],
                     v["max_reformulated_value"]))
        print("identity holds everywhere:", ci["identity_holds_everywhere"])
        for v in ("widgets", "root"):
            r = s[v]
            print("observed on %d held-out screenshots (%s): %d IDX nodes, values [%d, %d], "
                  "addresses [%d, %d], a+2 max %d"
                  % (r["screenshots"], v, r["n_idx_nodes"], r["observed_min"], r["observed_max"],
                     r["address_min"], r["address_max"], r["derived_a_plus_2_max"]))
            print("   corner calls %d, rect calls %d" % (r["corner_calls"], r["rect_calls"]))
    if st:
        rule("1b. SOUNDNESS -- exhaustive static range of every address-typed node")
        for k in sorted(st):
            r = st[k]
            print("%-26s coverage=%s nodes=%d reachable=[%d, %d]  (0,3071) sound=%s  "
                  "(0,3069) sound=%s"
                  % (k, r["coverage_complete"], r["address_nodes_declared"],
                     r["reachable_min"], r["reachable_max"],
                     r["sound_for_candidate_bound"], r["sound_for_3069_bound"]))
            print("     largest:", ", ".join("%s=%d" % (n, v) for n, v in r["six_largest_nodes"]))

    # ---------------------------------------------------------------- gates
    for name, label in (("gate.json", "A1, B"), ("gate_inline.json", "B+")):
        g = load(name)
        if not g:
            continue
        rule("2. GATE (%s) -- %s" % (name, g["gate_domain"]))
        print("reference arm %s, sha %s" % (g["reference_arm"], g["reference_source_sha256"][:12]))
        for arm, a in sorted(g["arms"].items()):
            print("  %-3s gate A %s %s   gate B %s (%d checked)   digests equal: %s"
                  % (arm, "PASS" if a["gate_A_passed"] else "FAIL",
                     a["gate_A_mismatch_fraction"], "PASS" if a["gate_B_passed"] else "FAIL",
                     a["gate_B_checked"],
                     a["gate_A_digest_reference"] == a["gate_A_digest_arm"]))
            print("      guards:", json.dumps({k: v for k, v in a["guards"].items() if v}))
        print("all_passed:", g["all_passed"])

    # ---------------------------------------------------------------- step 2
    runs = [load("measure_run%d.json" % i) for i in (1, 2, 3)]
    runs = [r for r in runs if r]
    if runs:
        rule("3. THREE COST MEASURES")
        arms = runs[0]["arms"]
        print("null-control pairs (byte-identical source):", runs[0]["null_control_pairs"])
        print("\nexecuted primitives (Registry.exact, module calls excluded):")
        for a in arms:
            p = runs[0]["primitives"][a]
            print("   %-3s worst %d  expected %.1f" % (a, p["worst_ops"], p["expected_ops"]))
        print("\nexecuted bytecodes per screenshot (exact, deterministic):")
        for a in arms:
            b = runs[0]["bytecodes"][a]
            print("   %-3s expected %12.1f   worst %12d   A0/x %.4f"
                  % (a, b["expected"], b["worst"], runs[0]["bytecode_ratios_A0_over_arm"][a]))
        print("\nwall clock, batch-one warm, us per screenshot:")
        print("   %-3s %-5s %12s %8s %s" % ("arm", "run", "median_us", "A0/x", "95% CI"))
        for a in arms:
            for i, r in enumerate(runs, 1):
                x = r["ratios_A0_over_arm"][a]
                print("   %-3s run%-2d %12.2f %8.4f [%.4f, %.4f]%s"
                      % (a, i, r["latency_us"][a]["median_us"], x["ratio"],
                         x["ci95_lo"], x["ci95_hi"], "  contains 1" if x["contains_1"] else ""))
        print("\nsource bytes:", json.dumps(runs[0]["source_bytes"]))

    # ---------------------------------------------------------------- where
    at = load("attribute.json")
    if at:
        rule("4. ATTRIBUTION -- by emitted function, one screenshot (seed 200)")
        for a in ("A0", "A1", "B", "B+"):
            if a not in at:
                continue
            r = at[a]
            print("%-3s calls %s" % (a, json.dumps(r["calls_per_screenshot"])))
            print("    kept %s" % json.dumps(r["guards_kept_by_function"]))
            print("    gone %s" % json.dumps(r["guards_eliminated_by_function"]))
            print("    bytecodes %s" % json.dumps(r["bytecodes_by_code_object"]))

    # ---------------------------------------------------------------- step 3
    sp = load("certificates_space.json")
    if sp:
        rule("5. CERTIFICATES -- search space")
        for k in sorted(sp):
            print("   %-24s %s" % (k, json.dumps(sp[k])))
    se = load("certificates_search.json")
    if se:
        rule("5b. CERTIFICATES -- the exhaustive sweeps")
        for k in sorted(se):
            for tag in ("s0", "s1", "s2"):
                if tag in se[k]:
                    v = se[k][tag]
                    print("   %-24s %s space %s evaluated %s conforming %s cert %s "
                          "exhausted %s unique %s"
                          % (k, tag, v["space_size"], v["evaluated"], v["conforming"],
                             v["certificate"], v["exhausted"], v["unique"]))
    pa = load("certificates_parse.json")
    if pa:
        rule("5c. CERTIFICATES -- section 33's parse, 12 held-out screens")
        for k in ("declared_IDX", "bounded"):
            if k in pa:
                r = pa[k]
                print("   %-13s rects %d/%d exact on %d/%d screens, roots %d, links %d/%d "
                      "(wrong %d), exact trees %d/%d"
                      % (k, r["rects_predicted"], r["widgets_in_probe"],
                         r["screens_rects_exact"], r["screens"], r["roots_predicted"],
                         r["parent_links_correct"], r["widgets_in_probe"],
                         r["parent_links_wrong"], r["trees_exact"], r["screens"]))


if __name__ == "__main__":
    main()
