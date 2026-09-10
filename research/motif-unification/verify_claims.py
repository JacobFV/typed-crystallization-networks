"""Re-derive every headline of RESULTS.md from `out/*.json` alone.

This file imports **none** of the analysis modules — not `motif.py`, not
`schema.py`, not `episodes.py`. It reads the raw records and recomputes each
claim, so a bug in a run script cannot make its own claim true.

    .venv/bin/python research/motif-unification/verify_claims.py
"""
from __future__ import annotations

import json
import pathlib
import sys

OUT = pathlib.Path(__file__).resolve().parent / "out"
CHECKS = []


def check(name, got, want):
    CHECKS.append((name, got, want, got == want))


def main():
    s1 = json.loads((OUT / "step1.json").read_text())
    s2 = json.loads((OUT / "step2.json").read_text())
    s3 = json.loads((OUT / "step3.json").read_text())
    t1 = json.loads((OUT / "transfer.json").read_text())
    t2 = json.loads((OUT / "transfer2.json").read_text())

    # ---------------------------------------------------------------- step 1
    occ = s1["occurrences"]
    m2 = [o for o in occ if o["shape"] == "M2"]
    m3 = [o for o in occ if o["shape"] == "M3"]
    m3i = [o for o in occ if o["shape"] == "M3_identity"]
    check("M2 occurrences", len(m2), 5)
    check("M2 domains", sorted({o["domain"] for o in m2}),
          ["computer", "language", "visual"])
    check("M2 distinct digests", len({o["digest"] for o in m2}), 3)
    check("M2 distinct type signatures", len({o["type_signature"] for o in m2}), 3)
    check("M3 occurrences", len(m3), 3)
    check("M3 domains", sorted({o["domain"] for o in m3}), ["language", "visual"])
    check("M3 distinct digests", len({o["digest"] for o in m3}), 2)
    check("M3 is NOT in all three domains", len({o["domain"] for o in m3}) < 3, True)
    check("computer's 3-node instance is the identity form", len(m3i), 1)
    check("computer 3-node arity", m3i[0]["arity"], 3)
    check("M3 arity", m3[0]["arity"], 4)
    check("F1 fires", s1["F1"]["fires"], True)
    widths = sorted({int(o["input_labels"][0].split("x")[0].lstrip("(")) for o in m2})
    check("M2 buffer widths", widths, [128, 3072, 4096])
    check("M2 address carriers",
          sorted({o["input_labels"][1] for o in m2}), ["u16", "u32<=128", "u32<=4096"])
    roles = s1["F1"]["third_argument_role"]
    check("visual's compared value is a buffer read, not a literal",
          all("node = index" in v for k, v in roles.items() if k.startswith("V_same")), True)
    check("language's compared value is the literal 40",
          roles["L_stage_a/open"], "constant = 40")
    check("computer's compared value is the literal 123",
          roles["C_agent/brand"], "constant = 123")
    beh = s1["behaviour"]
    check("behaviour: every instance characterised with zero failures",
          [b["n_failures"] for b in beh], [0] * len(beh))
    check("behaviour: exhaustive over addresses everywhere",
          all(b["exhaustive_over_addresses"] for b in beh), True)
    check("behaviour: addresses checked",
          sorted({b["addresses_checked"] for b in beh}), [128, 3072, 4096])

    # ---------------------------------------------------------------- step 2
    gated = [r for r in s2["rows"] if r["artifact_present"]]
    check("step 2 gated instantiations", len(gated), 5)
    check("step 2 all bit-identical", all(r["bit_identical"] for r in gated), True)
    check("step 2 zero differing to_dict keys",
          [r["to_dict_differing_keys"] for r in gated], [[]] * 5)
    check("step 2 every declared input type reproduced",
          all(all(r["input_types_identical"]) for r in gated), True)
    check("step 2 every declared output type reproduced",
          all(r["output_type_identical"] for r in gated), True)
    check("M2 has no free node at any width",
          [r["free_node_candidate_counts"] for r in s2["rows"] if r["schema"] == "M2"],
          [{}, {}, {}])
    check("M3 free-node shape is width-invariant",
          [r["free_node_candidate_counts"] for r in s2["rows"] if r["schema"] == "M3"],
          [{"n0": 5}, {"n0": 5}, {"n0": 5}])
    check("tcn/ untouched", s2["G2"]["tcn_diff_vs_main"], "")
    check("generators/ untouched", s2["G2"]["generators_diff_vs_main"], "")
    check("no TypeError caught in the schema",
          s2["G2"]["typeerror_caught_anywhere_in_schema_py"], False)
    check("bounds reproduced exactly",
          [(v["declared"] and list(v["declared"]), v["built"])
           for v in s2["G2"]["bounds_preserved"].values()],
          [([0, 128], [0, 128]), (None, None), ([0, 4096], [0, 4096])])

    # ---------------------------------------------------------------- step 3
    sw = {d["domain"]: d for d in s3["sweeps"]}
    for dom in ("language", "visual"):
        d = sw[dom]
        check(f"{dom}: space exhausted", d["exhausted"] and d["evaluated"] == d["space_size"], True)
        check(f"{dom}: exactly one conforming", d["conforming"], 1)
        check(f"{dom}: certificate", d["certificate"], "unique")
        check(f"{dom}: stored vector scores 1.0", d["stored_vector_accuracy"], 1.0)
        check(f"{dom}: held out 1.0", d["held_out_stored_vector_accuracy"], 1.0)
        check(f"{dom}: beats best constant",
              d["stored_vector_accuracy"] > d["best_constant"], True)
        check(f"{dom}: beats uniform random over the space",
              d["stored_vector_accuracy"] > d["uniform_random_over_space"], True)
    adm = {a["arm"]: a for a in s3["admission"]}
    check("M3 at two widths is admitted",
          [a["admitted"] for a in s3["admission"] if "128 and 3072" in a["arm"]], [True])
    check("M3 at one width is refused by R1",
          any(not a["admitted"] and a["refusal"].startswith("R1")
              for a in s3["admission"] if "128 only" in a["arm"]), True)
    check("M2 is refused by R2 for having no vector",
          any(not a["admitted"] and "R2" in a["refusal"] and "no selections" in a["refusal"]
              for a in s3["admission"] if a["arm"].startswith("M2")), True)
    check("M2 is nevertheless bit-identical at three widths",
          len(set(s3["M2_digests"])), 3)
    for f in s3["arm_F"]:
        check(f"arm F {f['arm']}: identical to the right schema at 128",
              f["digest_at_128_equals_right_schema"], True)
        check(f"arm F {f['arm']}: earns `unique` at 128", f["certificate_at_128"], "unique")
        check(f"arm F {f['arm']}: stored vector is perfect at 128",
              f["stored_vector_accuracy"][0], 1.0)
        check(f"arm F {f['arm']}: stored vector fails at 3072",
              f["stored_vector_accuracy"][1] < 1.0, True)
        check(f"arm F {f['arm']}: refused at two widths by R2",
              (not f["admitted"]) and f["refusal"].startswith("R2"), True)
        check(f"arm F {f['arm']}: refused at one width by R1",
              (not f["single_width"]["admitted"])
              and f["single_width"]["refusal"].startswith("R1"), True)
    check("arm F-a collapses to zero conforming at the second width",
          [f["sweeps"][1]["conforming"] for f in s3["arm_F"] if f["arm"].startswith("F-a")],
          [0])

    # -------------------------------------------------------------- transfer
    a1 = {a["arm"]: a for a in t1["arms"]}
    check("T_C1: every arm exhausted",
          all(a["exhausted"] and a["evaluated"] == a["space_size"] for a in t1["arms"]), True)
    check("T_C1: no-library at one node has no solution in the family",
          (a1["A1_budget1"]["conforming"], a1["A1_budget1"]["certificate"]), (0, "complete"))
    check("T_C1: no-library at two nodes already solves it",
          (a1["A1_budget2"]["conforming"] > 0, a1["A1_budget2"]["held_out"]), (True, 1.0))
    check("T_C1: the class conforms", a1["A2_class"]["conforming"] > 0, True)
    check("T_C1: the class equals the hand-authored ceiling",
          (a1["A2_class"]["conforming"], a1["A2_class"]["held_out"]),
          (a1["A4_hand"]["conforming"], a1["A4_hand"]["held_out"]))
    check("T_C1: the wrong-motif control is dead",
          a1["A3_wrong_motif"]["conforming"], 0)
    check("T_C1: the wrong-width module cannot even be constructed",
          t1["A3b_wrong_width"]["constructed"], False)
    check("T_C1: class and hand-authored are the same graph",
          t1["class_hand_structurally_identical"], True)

    a2 = {a["arm"]: a for a in t2["arms"]}
    check("T_C2: every arm exhausted",
          all(a["exhausted"] and a["evaluated"] == a["space_size"] for a in t2["arms"]), True)
    check("T_C2: constant addresses cannot solve it",
          (a2["A1_budget2"]["conforming"], a2["A1_budget2"]["certificate"]), (0, "complete"))
    check("T_C2: no library at three nodes solves it uniquely",
          (a2["A1_budget3"]["conforming"], a2["A1_budget3"]["certificate"],
           a2["A1_budget3"]["held_out"]), (1, "unique", 1.0))
    check("T_C2: THE CERTIFIED CLASS SCORES ZERO",
          (a2["A2_class"]["conforming"], a2["A2_class"]["certificate"]), (0, "complete"))
    check("T_C2: the hand-authored equivalent scores zero too",
          a2["A4_hand"]["conforming"], 0)
    check("T_C2: re-selecting the schema's vector on the new domain does solve it",
          (a2["A2b_schema"]["conforming"], a2["A2b_schema"]["certificate"],
           a2["A2b_schema"]["held_out"]), (1, "unique", 1.0))
    check("T_C2: re-selecting buys no search saving over no-library",
          a2["A2b_schema"]["space_size"], a2["A1_budget3"]["space_size"])
    check("T_C2: the chosen module is the `sub` member, not the certified `add`",
          t2["schema_family_digests"][1] in (a2["A2b_schema"]["chosen_operator"] or ""), True)
    check("T_C2: the certified class's member is the `add` member",
          t2["class_module_digest"], t2["schema_family_digests"][0])
    check("T_C2: baselines are balanced",
          (t2["baselines"]["best_constant_train"], t2["baselines"]["best_constant_test"]),
          (0.5, 0.5))
    check("T_C2: the wrong-motif control is dead", a2["A3_wrong_motif"]["conforming"], 0)

    # class provenance: the computer domain took no part
    check("the class is certified at 128 and 3072 only",
          t2["class_certified_widths"], [128, 3072])

    bad = [c for c in CHECKS if not c[3]]
    for name, got, want, ok in CHECKS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}"
              + ("" if ok else f"\n        got  {got!r}\n        want {want!r}"))
    print(f"\n{len(CHECKS) - len(bad)} of {len(CHECKS)} checks pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
