"""Stage 1 -- recover a lazy conditional that skips an expensive continuation.

Arms, all measured by the same instrument:

  spec              the frozen `Program`, eager, as the algebra defines it
  control_lazy      the same program transcribed 1:1 into the algorithm IR and
                    run under a call-by-need evaluator, with no construct added.
                    This is the control that keeps any gain attributable to the
                    `Guard` construct and not to the evaluation strategy.
  resynth           whatever `synth.rewrite_search` finds
  ablation_noguard  the same search with `Guard` removed from the grammar
  supplied          the same search with the target expression handed to it
                    verbatim as a candidate -- the template-instantiation control
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import ablate
import cost
import ir
import runner
import spec as SPEC
import synth

OUT = HERE / "out"
DOM = SPEC.sparse_guard_domain


def hit_class(case, hit):
    return tuple(int(v == hit) for v in case["x"])


def main(n=4, hit=15, tag="stage1", max_size=4, sample=128, verify_sample=None,
         run_ablations=True, expensive=None):
    fx = SPEC.sparse_guard(n=n, hit=hit) if expensive is None else expensive
    report = {"miniature": "sparse_guard", "n": n, "hit": hit,
              "domain_size": len(list(DOM(fx))),
              "spec_digest": fx["program"].digest,
              "spec_nodes": len(fx["program"].nodes),
              "spec_execution_cost_scalar": fx["program"].execution_cost(fx["registry"])}

    print("== spec profile")
    sp, spec_outs = runner.spec_profile(fx, DOM)
    report["spec"] = sp

    print("== control: call-by-need evaluator over the unmodified specification")
    control = ir.from_program(fx["program"], fx["registry"])
    cp, _ = runner.alg_profile(fx, control, DOM)
    report["control_lazy_evaluator"] = cp

    print("== resynthesis (full IR)")
    res = synth.rewrite_search(fx, max_size=max_size, sample=sample, domain=DOM,
                               verify_sample=verify_sample)
    prog = res["program"]
    report["search"] = {k: v for k, v in res.items() if k != "program"}
    report["constructs_used"] = sorted(prog.constructs())
    report["result_bindings"] = ["%s = %r" % (k, v) for k, v in prog.bindings]

    ap, alg_outs = runner.alg_profile(fx, prog, DOM)
    report["resynth"] = ap
    report["ratios"] = cost.ratio_table(sp, ap)

    print("== certificate")
    report["certificate"] = runner.certify(fx, prog, DOM)
    report["certificate"]["outputs_agree_digest"] = (sp["output_digest"] == ap["output_digest"])

    print("== bytecodes")
    mod, cres = runner.compiled_spec_module(fx)
    src = ir.Lowerer(fx["registry"]).program(prog, entry="run")
    gen = ir.load(src)
    cases = list(DOM(fx))
    by_class = {}
    for c in cases:
        by_class.setdefault(hit_class(c, hit), []).append(c)
    weights = {k: len(v) / len(cases) for k, v in by_class.items()}
    probe = {k: v[:2] for k, v in by_class.items()}
    spec_bc = runner.bytecode_classes(lambda c: mod.run({"x": c["x"]}, validate=False), probe)
    alg_bc = runner.bytecode_classes(lambda c: gen.run(c["x"]), probe)
    report["bytecodes"] = {
        "classes": len(by_class),
        "class_definition": "the pattern of which positions satisfy the predicate",
        "spec_compiled_expected": runner.expected_bytecodes(spec_bc, weights),
        "spec_compiled_worst": runner.worst_bytecodes(spec_bc),
        "resynth_expected": runner.expected_bytecodes(alg_bc, weights),
        "resynth_worst": runner.worst_bytecodes(alg_bc),
        "spec_per_class": spec_bc, "resynth_per_class": alg_bc,
        "class_weights": {str(k): v for k, v in weights.items()},
    }
    report["bytecodes"]["expected_ratio"] = (report["bytecodes"]["spec_compiled_expected"]
                                            / report["bytecodes"]["resynth_expected"])
    report["bytecodes"]["worst_ratio"] = (report["bytecodes"]["spec_compiled_worst"]
                                          / report["bytecodes"]["resynth_worst"])

    print("== deployment")
    report["deployment"] = runner.deploy(fx, prog, tag, DOM, spec_outs)

    if run_ablations:
        print("== ablation matrix (criterion 3)")
        rows = ablate.matrix(fx, DOM, synth.rewrite_search, sp["expected_ops"],
                             max_size=max_size, sample=sample,
                             verify_sample=verify_sample)
        report["ablation_matrix"] = rows
        report["ablation_summary"] = ablate.summarize(rows)

        print("== control: the target supplied verbatim to a Guard-less grammar")
        seeds = [v for k, v in prog.bindings if isinstance(v, ir.Guard)]
        sup = synth.rewrite_search(fx, grammar_constructs=("Find", "Scan", "Let"),
                                   max_size=max_size, sample=sample, domain=DOM,
                                   seed_candidates=seeds, verify_sample=verify_sample,
                                   verbose=False)
        supp, _ = runner.alg_profile(fx, sup["program"], DOM)
        report["control_target_supplied"] = {
            "seeded_with": [repr(s) for s in seeds],
            "profile": supp,
            "expected_ratio_vs_spec": sp["expected_ops"] / supp["expected_ops"],
            "note": "this row is template instantiation by construction; it is "
                    "reported so it is distinguishable from the unseeded run"}

    (OUT / ("%s.json" % tag)).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in
                      ("spec", "control_lazy_evaluator", "resynth", "ratios",
                       "certificate", "bytecodes", "deployment") if k in report},
                     indent=2, default=str)[:3000])
    return report


if __name__ == "__main__":
    main()
