"""Criterion 4 -- structural cases not used during synthesis.

The exhaustive certificate of stages 1 and 2 already covers every input of their
declared domains, so "held out" cannot mean more inputs of the same problem.
What is held out here is *structure*: a different number of positions, a
different carrier width, a different hit constant, and a different expensive or
predicate module.  The engines are re-run unchanged on each, and each result is
certified exhaustively against the typed interpreter.

This also answers a question the pre-registration does not ask but which decides
whether any of this generalizes: does the search recover the *same* construct on
a problem it has not seen, or did stage 1 and stage 2 happen to work?
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import cost
import ir
import runner
import spec as SPEC
import synth

OUT = HERE / "out"

GUARD_VARIANTS = [
    {"n": 3, "hit": 15, "bits": 4},
    {"n": 5, "hit": 7, "bits": 3},
    {"n": 4, "hit": 2, "bits": 3, "coeffs": ((3, 1), (5, 2), (7, 4))},
    {"n": 4, "hit": 0, "bits": 4, "coeffs": ((2, 1),)},
]

FIND_VARIANTS = [
    {"n": 4},
    {"n": 6},
    {"n": 6, "a": 3, "b": 1, "m": 5, "thr": 2},
    {"n": 7, "a": 7, "b": 2, "m": 11, "thr": 6},
]


def main(tag="heldout"):
    rows = {"sparse_guard": [], "find_first": []}

    for kw in GUARD_VARIANTS:
        fx = SPEC.sparse_guard(**kw)
        dom = SPEC.sparse_guard_domain
        size = len(fx["carrier_values"]) ** fx["n"]
        print("== sparse_guard %r  domain %d" % (kw, size))
        sp, _ = runner.spec_profile(fx, dom)
        res = synth.rewrite_search(fx, max_size=4, sample=128, domain=dom, verbose=False)
        prog = res["program"]
        ap, _ = runner.alg_profile(fx, prog, dom)
        cert = runner.certify(fx, prog, dom)
        rows["sparse_guard"].append({
            "variant": kw, "domain": size,
            "spec_expected": sp["expected_ops"], "spec_worst": sp["worst_ops"],
            "alg_expected": ap["expected_ops"], "alg_worst": ap["worst_ops"],
            "expected_ratio": sp["expected_ops"] / ap["expected_ops"],
            "worst_ratio": sp["worst_ops"] / ap["worst_ops"],
            "constructs_used": sorted(prog.constructs()),
            "guards_introduced": sum(1 for _, e in prog.bindings if isinstance(e, ir.Guard)),
            "certificate": cert,
            "counterexamples_during_search": res["counterexamples_found"]})
        print("   ", json.dumps(rows["sparse_guard"][-1]["expected_ratio"]),
              "guards", rows["sparse_guard"][-1]["guards_introduced"],
              "exact", cert["exact"])

    for kw in FIND_VARIANTS:
        fx = SPEC.find_first(**kw)
        dom = SPEC.find_first_domain
        size = 4 ** fx["n"]
        print("== find_first %r  domain %d" % (kw, size))
        sp, _ = runner.spec_profile(fx, dom)
        res = synth.toplevel_search(fx, domain=dom, sample=128, body_size=3,
                                    tail_size=2, verbose=False)
        prog = res["program"]
        if prog is None:
            rows["find_first"].append({"variant": kw, "domain": size, "solved": False})
            print("    NOTHING FOUND")
            continue
        ap, _ = runner.alg_profile(fx, prog, dom)
        cert = runner.certify(fx, prog, dom)
        rows["find_first"].append({
            "variant": kw, "domain": size, "solved": True,
            "result": repr(prog.output),
            "spec_expected": sp["expected_ops"], "spec_worst": sp["worst_ops"],
            "alg_expected": ap["expected_ops"], "alg_worst": ap["worst_ops"],
            "expected_ratio": sp["expected_ops"] / ap["expected_ops"],
            "worst_ratio": sp["worst_ops"] / ap["worst_ops"],
            "constructs_used": sorted(prog.constructs()),
            "certificate": cert})
        print("   ", rows["find_first"][-1]["expected_ratio"], "exact", cert["exact"])

    (OUT / ("%s.json" % tag)).write_text(json.dumps(rows, indent=2, default=str))
    return rows


if __name__ == "__main__":
    main()
