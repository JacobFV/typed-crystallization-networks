"""The §41 cross-check: does the winning objective also stop picking the
bytecode-maximal program?

This is a cross-check, not a new experiment.  `research/program-length/out/
language_family.json` is *read*; nothing is re-enumerated.  It holds the ten
conforming programs of §41's language family (45 375 evaluated, `exhausted:
true`), all at unseen accuracy 1.000 against a 0.5661 majority constant, with
`description_bits ∈ {4 043 552, 4 043 560, 4 043 568}` and measured
`bytecodes_total ∈ {41 417, 41 441, 41 465}`.

Each objective is applied in the form it takes on a **single-task** family, and
where it degenerates that is stated rather than papered over.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "research" / "program-length" / "out" / "language_family.json"
OUT = Path(__file__).resolve().parent / "out"


def main():
    d = json.loads(SRC.read_text())
    rows = d["rows"]
    for i, r in enumerate(rows):
        r["_i"] = i
    base = max(r["description_bits"] for r in rows)          # the worst description
    tasks = 1                                                # one language family

    def pick(key, rs):
        return min(rs, key=key)

    res = {}
    # O1 -- incumbent: maximise saving == minimise description_bits.
    o1 = pick(lambda r: (-(base - r["description_bits"]), r["_i"]), rows)
    res["O1"] = {"degenerate": False, "note": "maximise saving = minimise description_bits"}
    # O2 -- breadth-weighted: saving x |T|.  |T| = 1 for every candidate.
    o2 = pick(lambda r: (-(base - r["description_bits"]) * tasks, r["_i"]), rows)
    res["O2"] = {"degenerate": True,
                 "note": "|T| = 1 for every candidate, so the weight is a constant "
                         "multiplier and the ordering is identical to O1"}
    # O3 -- per-task mean over one task: the mean is the value.
    o3 = pick(lambda r: (-(base - r["description_bits"]) / tasks, r["_i"]), rows)
    res["O3"] = {"degenerate": True,
                 "note": "one task, so the per-task mean is the value itself"}
    # O4 -- in-corpus leave-one-out: holding out the only task leaves nothing to
    # mine from, so every candidate scores 0 and the objective is undefined.
    res["O4"] = {"degenerate": True, "pick": None, "note":
                 "holding out the only task empties the mining corpus; every "
                 "candidate scores 0 and the objective carries no signal at all"}
    # O5 -- measured execution cost: minimise measured executed bytecodes.
    o5 = pick(lambda r: (r["bytecodes_total"], r["_i"]), rows)
    res["O5"] = {"degenerate": False,
                 "note": "minimise measured executed bytecodes (§41's own instrument)"}
    # B1/B2 -- frequency counts: every candidate solves the one task once.
    res["B1"] = {"degenerate": True, "pick": None,
                 "note": "every candidate covers the single task; a complete tie"}
    res["B2"] = dict(res["B1"])

    for oid, r in (("O1", o1), ("O2", o2), ("O3", o3), ("O5", o5)):
        res[oid]["pick"] = {"description_bits": r["description_bits"],
                            "bytecodes_total": r["bytecodes_total"],
                            "execution_cost": r["execution_cost"],
                            "nodes": r["nodes"],
                            "unseen_accuracy": r["heldout_unseen_lengths"]["accuracy"],
                            "unseen_majority_constant":
                                r["heldout_unseen_lengths"]["majority_constant"],
                            "is_bytecode_maximal":
                                r["bytecodes_total"] == max(x["bytecodes_total"] for x in rows),
                            "is_bytecode_minimal":
                                r["bytecodes_total"] == min(x["bytecodes_total"] for x in rows),
                            "is_description_minimal":
                                r["description_bits"] == min(x["description_bits"] for x in rows)}

    payload = {"source": str(SRC.relative_to(ROOT)),
               "enumeration": d["enumeration"],
               "distinct_description_bits": d["distinct_description_bits"],
               "distinct_bytecodes": d["distinct_bytecodes"],
               "distinct_execution_cost": d["distinct_execution_cost"],
               "n_rows": len(rows),
               "description_vs_bytecodes": sorted(
                   {(r["description_bits"], r["bytecodes_total"]) for r in rows}),
               "objectives": res}
    OUT.mkdir(exist_ok=True)
    (OUT / "s41_crosscheck.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    for oid in ("O1", "O2", "O3", "O4", "O5", "B1", "B2"):
        v = res[oid]
        p = v.get("pick")
        print(f"{oid}  degenerate={v['degenerate']:1}  " +
              ("no pick (tie / undefined)" if p is None else
               f"desc={p['description_bits']} bytecodes={p['bytecodes_total']} "
               f"bytecode_maximal={p['is_bytecode_maximal']} "
               f"bytecode_minimal={p['is_bytecode_minimal']}"))
    print("desc -> bytecodes:", payload["description_vs_bytecodes"])
    print("->", OUT / "s41_crosscheck.json")


if __name__ == "__main__":
    main()
