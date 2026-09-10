"""Do FINDINGS section 49's bytecode numbers reproduce here?

Section 49 records, for the two miniatures, executed CPython bytecodes:

    stage 1   compiled specification 1003.25 expected / 1007 worst
              resynthesized           157.00 expected / 1507 worst   -> 0.67x worst
    stage 2   compiled specification  821.00 expected /  821 worst
              resynthesized           338.76 expected / 1303 worst   -> 0.63x worst

Those expectations are exact class-weighted expectations over the whole 65,536
domain, not samples, so they are reproduced the same way here: every input is
counted.  Bytecode counts do not depend on machine load, so this check is valid
whatever else is running.

The `bare` entry is used, because section 49 counted the generated function, not a
dict envelope around it.
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "research" / "lazy-guard"))

import arms                     # noqa: E402
import cost as lazy_cost        # noqa: E402

RECORDED = {
    1: {"spec_expected": 1003.25, "spec_worst": 1007,
        "resynth_expected": 157.00, "resynth_worst": 1507, "worst_ratio": 0.67},
    2: {"spec_expected": 821.00, "spec_worst": 821,
        "resynth_expected": 338.76, "resynth_worst": 1303, "worst_ratio": 0.63},
}


def counts(call, recs):
    xs = [lazy_cost.count_opcodes(lambda r=r: call(r)) for r in recs]
    return {"expected": statistics.mean(xs), "worst": max(xs), "best": min(xs),
            "cases": len(xs)}


def main(stages=(1, 2)):
    out = {}
    for stage in stages:
        print("== stage %d, all %d inputs" % (stage, 65536))
        g = arms.build(stage)
        recs = g["uniform"]
        a = counts(g["arms"]["A"]["call"], recs)      # arm A has no bare entry
        b = counts(g["arms"]["B"]["bare"], recs)
        rec = RECORDED[stage]
        row = {"cases": len(recs), "A_envelope": a, "B_bare": b,
               "worst_ratio_A_over_B": a["worst"] / b["worst"],
               "expected_ratio_A_over_B": a["expected"] / b["expected"],
               "findings_49_recorded": rec,
               "worst_ratio_reproduces_to_2dp":
                   round(a["worst"] / b["worst"], 2) == rec["worst_ratio"]}
        out["stage%d" % stage] = row
        print("   A  expected %8.2f  worst %5d   (section 49: %8.2f / %d)"
              % (a["expected"], a["worst"], rec["spec_expected"], rec["spec_worst"]))
        print("   B  expected %8.2f  worst %5d   (section 49: %8.2f / %d)"
              % (b["expected"], b["worst"], rec["resynth_expected"], rec["resynth_worst"]))
        print("   worst ratio A/B = %.3f  (section 49: %.2f)  reproduces: %s"
              % (row["worst_ratio_A_over_B"], rec["worst_ratio"],
                 row["worst_ratio_reproduces_to_2dp"]))
    (arms.OUT / "bytecode_recheck.json").write_text(json.dumps(out, indent=1))
    print("-> %s" % (arms.OUT / "bytecode_recheck.json"))
    return out


if __name__ == "__main__":
    main()
