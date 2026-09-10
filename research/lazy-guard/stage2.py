"""Stage 2 -- recover find-first / a bounded scan from an unrolled family.

The specification evaluates all N predicates and folds them with a `mux` chain,
which is the only way the algebra can express "the first index that satisfies
p".  The resynthesizer must (a) notice that the N predicate sub-DAGs are one
parametric family, which is anti-unification, and (b) compose a construct that
stops at the first hit.  Neither step is a local rewrite of any node, which is
what separates this stage from stage 1.
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
DOM = SPEC.find_first_domain


def main(n=8, tag="stage2", body_size=3, tail_size=2, sample=128, run_ablations=True):
    fx = SPEC.find_first(n=n)
    p, r = fx["program"], fx["registry"]
    report = {"miniature": "find_first", "n": n,
              "domain_size": len(list(DOM(fx))),
              "spec_digest": p.digest, "spec_nodes": len(p.nodes),
              "spec_execution_cost_scalar": p.execution_cost(r)}

    print("== spec profile")
    sp, spec_outs = runner.spec_profile(fx, DOM)
    report["spec"] = sp

    print("== control: call-by-need evaluator over the unmodified specification")
    control = ir.from_program(p, r)
    cp, _ = runner.alg_profile(fx, control, DOM)
    report["control_lazy_evaluator"] = cp

    print("== anti-unification + top-level resynthesis")
    res = synth.toplevel_search(fx, domain=DOM, sample=sample,
                                body_size=body_size, tail_size=tail_size)
    prog = res["program"]
    report["search"] = {k: v for k, v in res.items() if k != "program"}
    if prog is None:
        report["verdict"] = "no exact candidate in the declared space"
        (OUT / ("%s.json" % tag)).write_text(json.dumps(report, indent=2, default=str))
        return report
    report["result"] = repr(prog.output)
    report["constructs_used"] = sorted(prog.constructs())

    ap, alg_outs = runner.alg_profile(fx, prog, DOM)
    report["resynth"] = ap
    report["ratios"] = cost.ratio_table(sp, ap)

    print("== certificate")
    report["certificate"] = runner.certify(fx, prog, DOM)
    report["certificate"]["outputs_agree_digest"] = (sp["output_digest"] == ap["output_digest"])

    print("== bytecodes")
    mod, _ = runner.compiled_spec_module(fx)
    src = ir.Lowerer(r).program(prog, entry="run")
    gen = ir.load(src)
    cases = list(DOM(fx))

    def first_true(case):
        return next(iter(gen.run(case["y"]) for _ in (0,)))
    by_class = {}
    for c in cases:
        by_class.setdefault(gen.run(c["y"]), []).append(c)
    weights = {k: len(v) / len(cases) for k, v in by_class.items()}
    probe = {k: v[:2] for k, v in by_class.items()}
    spec_bc = runner.bytecode_classes(lambda c: mod.run({"y": c["y"]}, validate=False), probe)
    alg_bc = runner.bytecode_classes(lambda c: gen.run(c["y"]), probe)
    report["bytecodes"] = {
        "classes": len(by_class),
        "class_definition": "the index of the first satisfying position (N if none)",
        "spec_compiled_expected": runner.expected_bytecodes(spec_bc, weights),
        "spec_compiled_worst": runner.worst_bytecodes(spec_bc),
        "resynth_expected": runner.expected_bytecodes(alg_bc, weights),
        "resynth_worst": runner.worst_bytecodes(alg_bc),
        "spec_per_class": spec_bc, "resynth_per_class": alg_bc,
        "class_weights": {str(k): v for k, v in weights.items()}}
    report["bytecodes"]["expected_ratio"] = (report["bytecodes"]["spec_compiled_expected"]
                                            / report["bytecodes"]["resynth_expected"])
    report["bytecodes"]["worst_ratio"] = (report["bytecodes"]["spec_compiled_worst"]
                                          / report["bytecodes"]["resynth_worst"])

    print("== deployment")
    report["deployment"] = runner.deploy(fx, prog, tag, DOM, spec_outs)

    if run_ablations:
        print("== ablation matrix (criterion 3)")
        rows = ablate.matrix(fx, DOM, synth.toplevel_search, sp["expected_ops"],
                             sample=sample, body_size=body_size, tail_size=tail_size)
        report["ablation_matrix"] = rows
        report["ablation_summary"] = ablate.summarize(rows)

        print("== ablation: both data-dependent constructs removed")
        both = synth.toplevel_search(fx, constructs=("Guard", "Let"), domain=DOM,
                                     sample=sample, body_size=body_size,
                                     tail_size=tail_size, verbose=False)
        if both["program"] is None:
            report["ablation_no_find_no_scan"] = {"solved": False,
                                                  "candidates_considered": both["candidates_considered"]}
        else:
            bp, _ = runner.alg_profile(fx, both["program"], DOM)
            report["ablation_no_find_no_scan"] = {
                "solved": True, "profile": bp,
                "expected_ratio_vs_spec": sp["expected_ops"] / bp["expected_ops"],
                "result": repr(both["program"].output)}

        print("== control: the target supplied verbatim to a Find-less grammar")
        sup = synth.toplevel_search(fx, constructs=("Guard", "Let"), domain=DOM,
                                    sample=sample, body_size=body_size,
                                    tail_size=tail_size, verbose=False,
                                    seed_candidates=[prog.output])
        if sup["program"] is not None:
            supp, _ = runner.alg_profile(fx, sup["program"], DOM)
            report["control_target_supplied"] = {
                "seeded_with": repr(prog.output), "profile": supp,
                "expected_ratio_vs_spec": sp["expected_ops"] / supp["expected_ops"],
                "note": "template instantiation by construction; reported so the "
                        "unseeded run is distinguishable from it"}

    (OUT / ("%s.json" % tag)).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in
                      ("result", "spec", "control_lazy_evaluator", "resynth", "ratios",
                       "certificate", "deployment") if k in report}, indent=2, default=str)[:2500])
    return report


if __name__ == "__main__":
    main()
