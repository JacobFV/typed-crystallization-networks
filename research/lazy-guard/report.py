"""Print every number RESULTS.md quotes, straight from `out/*.json`.

The document is written from this output, so a headline in the prose can be
checked against the raw file it came from without re-running anything.
"""
from __future__ import annotations

import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "out"


def load(name):
    p = OUT / ("%s.json" % name)
    return json.loads(p.read_text()) if p.exists() else None


def line(*a):
    print(*a)


def main():
    inst = load("instrumentation")
    if inst:
        line("### instrumentation")
        for k, v in inst.items():
            line("  %-14s domain=%-7d certified=%-7d execution_cost=%-7s "
                 "spec ops worst/expected/best=%s/%s/%s  bins=%d  compiled bytecodes=%s"
                 % (k, v["domain_size"], v["native_evaluator_certified_on"],
                    v["execution_cost_scalar"], v["spec_worst_ops"],
                    v["spec_expected_ops"], v["spec_best_ops"],
                    len(v["spec_ops_histogram"]), v["spec_compiled_bytecodes"]))
            line("                 module leaf costs %s" % v["module_leaf_cost"])

    for tag in ("stage1", "stage2"):
        d = load(tag)
        if not d:
            continue
        line("")
        line("### %s  (%s, n=%s, domain=%s)" % (tag, d["miniature"], d["n"], d["domain_size"]))
        line("  spec           worst=%s expected=%s best=%s" %
             (d["spec"]["worst_ops"], d["spec"]["expected_ops"], d["spec"]["best_ops"]))
        c = d["control_lazy_evaluator"]
        line("  control lazy   worst=%s expected=%s best=%s   (no construct added)" %
             (c["worst_ops"], c["expected_ops"], c["best_ops"]))
        a = d["resynth"]
        line("  resynth        worst=%s expected=%s best=%s" %
             (a["worst_ops"], a["expected_ops"], a["best_ops"]))
        line("  ratios         expected=%.4f  worst=%.4f  best=%.4f" %
             (d["ratios"]["expected_ratio"], d["ratios"]["worst_ratio"], d["ratios"]["best_ratio"]))
        line("  constructs     %s" % d["constructs_used"])
        if "result" in d:
            line("  result         %s" % d["result"])
        if tag == "stage1":
            for t in d["search"]["trace"]:
                line("     rewrite     %-4s := %s" % (t["binding"], t["replacement"]))
            line("  search         %d candidates, %d rounds, %d counterexamples %s" %
                 (d["search"]["candidates_considered"], d["search"]["rounds"],
                  d["search"]["counterexamples_found"], d["search"]["counterexamples"]))
        else:
            s = d["search"]
            line("  search         %d candidates, %d exact on samples, %d rejected exhaustively" %
                 (s["candidates_considered"], s["exact_on_samples"], s["rejected_by_exhaustive_check"]))
            for cc in s["components"]:
                line("     component   x%-3s %s  [%s]" % (cc["arity"], cc["lambda"][:80], cc["abstracted"]))
        cert = d["certificate"]
        line("  certificate    exact=%s checked=%s digests_agree=%s oracle=%s" %
             (cert["exact"], cert["checked"], cert["outputs_agree_digest"], cert["oracle"]))
        b = d["bytecodes"]
        line("  bytecodes      spec expected=%.2f worst=%d | alg expected=%.2f worst=%d "
             "| ratios expected=%.3f worst=%.3f  (%d classes)" %
             (b["spec_compiled_expected"], b["spec_compiled_worst"],
              b["resynth_expected"], b["resynth_worst"],
              b["expected_ratio"], b["worst_ratio"], b["classes"]))
        dep = d["deployment"]
        line("  deployment     stdlib_only=%s bit_identical=%s cases=%s bytes=%d imports=%s" %
             (dep["stdlib_only"], dep["bit_identical"], dep["cases"],
              dep["source_bytes"], dep["detail"].get("modules")))
        if "ablation_matrix" in d:
            line("  ablation matrix:")
            for k, v in d["ablation_matrix"].items():
                line("     %-14s solved=%-5s expected=%-10s ratio=%-8s constructs=%s" %
                     (k, v["solved"], v["expected_ops"],
                      ("%.3f" % v["expected_ratio_vs_spec"]) if v["expected_ratio_vs_spec"] else "-",
                      v["constructs_used"]))
            line("     summary      %s" % d["ablation_summary"])
        if "ablation_no_find_no_scan" in d:
            line("     minus Find+Scan  %s" % json.dumps(d["ablation_no_find_no_scan"])[:200])
        if "control_target_supplied" in d:
            t = d["control_target_supplied"]
            line("     target supplied verbatim -> ratio %.3f  (%s)" %
                 (t["expected_ratio_vs_spec"], "template instantiation control"))

    d = load("stage3")
    if d:
        line("")
        line("### stage3  (%s, span=%s, rect nodes=%s)" % (d["subroutine"], d["span"], d["rect_nodes"]))
        line("  execution_cost scalar %s" % d["execution_cost_scalar"])
        line("  oracle chain   %s" % d["oracle_chain"])
        for k, v in d.get("result", {}).items():
            line("  %-9s -> %s" % (k, v[:190]))
        for key in ("search_domain", "heldout", "corner_distribution"):
            if key in d:
                v = d[key]
                line("  %-18s spec worst/expected = %s/%.2f | alg = %s/%.2f | ratios expected=%.3f worst=%.3f"
                     % (key, v["spec"]["worst_ops"], v["spec"]["expected_ops"],
                        v["resynth"]["worst_ops"], v["resynth"]["expected_ops"],
                        v["ratios"]["expected_ratio"], v["ratios"]["worst_ratio"]))
        if "heldout" in d and "certificate" in d["heldout"]:
            line("  heldout cert   %s" % json.dumps(d["heldout"]["certificate"])[:260])
        if "level_check" in d:
            line("  level check    %s" % json.dumps({k: v for k, v in d["level_check"].items()
                                                     if k != "reading"}))
        if "deployment" in d:
            line("  deployment     %s" % json.dumps({k: v for k, v in d["deployment"].items()
                                                     if k != "note"}))
        if "deployment_error" in d:
            line("  deployment err %s" % d["deployment_error"])
        if "ablation_matrix" in d:
            for k, v in d["ablation_matrix"].items():
                line("     %-14s solved=%-5s expected=%-10s ratio=%-8s constructs=%s" %
                     (k, v["solved"], v["expected_ops"],
                      ("%.3f" % v["expected_ratio_vs_spec"]) if v["expected_ratio_vs_spec"] else "-",
                      v["constructs_used"]))
            line("     summary      %s" % d["ablation_summary"])
        if "ablation_no_find_no_scan" in d:
            line("     minus Find+Scan  %s" % json.dumps(d["ablation_no_find_no_scan"])[:200])

    h = load("heldout")
    if h:
        line("")
        line("### held-out structural variants")
        for fam, rows in h.items():
            for r in rows:
                if not r.get("solved", True):
                    line("  %-13s %-52s NOT SOLVED" % (fam, r["variant"]))
                    continue
                line("  %-13s %-52s domain=%-8d expected %.2f -> %.2f = %.3fx | worst %.0f -> %.0f = %.3fx | exact=%s/%s | %s"
                     % (fam, r["variant"], r["domain"], r["spec_expected"], r["alg_expected"],
                        r["expected_ratio"], r["spec_worst"], r["alg_worst"], r["worst_ratio"],
                        r["certificate"]["exact"], r["certificate"]["checked"],
                        [c for c in r["constructs_used"] if c in ("Guard", "Find", "Scan")]))


if __name__ == "__main__":
    main()
