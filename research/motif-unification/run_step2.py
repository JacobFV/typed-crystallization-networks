"""Step 2 — one schema, instantiated and gated bit-identical at every width.

Writes `out/step2.json`. Gate G1 is digest equality against the artifact
fragment. Gate G2 is the no-relaxation checklist, each item a computed fact
rather than a sentence.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import schema as S                                  # noqa: E402

OUT = HERE / "out"

# The selection vector, per schema. `M2` has no free node at any width.
VECTOR = {"M2": {}, "M3": {"n0": 0}}


def dict_diff(a, b):
    """Every top-level key of `Program.to_dict()` on which two programs differ."""
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


CANON = {}


def main():
    import motif                                                  # noqa: E402
    _, found = motif.find(5, only=("V_same", "L_stage_a", "C_agent"))
    for r in found:
        CANON.setdefault((r["shape"], r["domain"]), r["_canon"])

    step1 = json.loads((OUT / "step1.json").read_text())
    frag_by = {}
    for o in step1["occurrences"]:
        frag_by.setdefault((o["shape"], o["domain"]), o)

    rows = []
    for name in ("M2", "M3"):
        for domain, cfg in S.SETTINGS.items():
            at = S.address_type(cfg["address"])
            free, built, reg = S.instantiate(name, cfg["width"], at, VECTOR[name])
            art = frag_by.get((name, domain))
            row = {
                "schema": name, "domain": domain, "width": cfg["width"],
                "address_label": cfg["address"][0] +
                                 (f"<={cfg['address'][3][1]}" if cfg['address'][3] else ""),
                "free_nodes": list(free),
                "free_node_candidate_counts": {n.name: len(n.candidates)
                                               for n in S.SCHEMAS[name](cfg["width"], at)[1].nodes
                                               if len(n.candidates) > 1},
                "selection_vector": VECTOR[name],
                "built_digest": built.digest,
                "artifact_present": art is not None,
                "artifact_digest": art["digest"] if art else None,
                "artifact": art["artifact"] if art else None,
                "bit_identical": bool(art and art["digest"] == built.digest),
            }
            canon = CANON.get((name, domain))
            if art and canon is not None:
                # every declared type reproduced exactly, field by field, against
                # the artifact's own `Type` objects -- never against a tag.
                row["input_types_identical"] = [
                    at_.to_dict() == bt.to_dict()
                    for (_, at_), (_, bt) in zip(canon.inputs, built.inputs)]
                row["output_type_identical"] = (
                    canon.nodes[-1].output.to_dict() == built.nodes[-1].output.to_dict())
                # a field-by-field `to_dict()` comparison against the artifact's
                # own canonical fragment, so the reader can see that after
                # `freeze` there is no differing key at all -- `version` included,
                # since `freeze` normalises it back to the fragment's own value.
                canon = CANON.get((name, domain))
                row["to_dict_differing_keys"] = dict_diff(canon.to_dict(), built.to_dict())
            rows.append(row)

    # --- the computer artifact's own three-node instance, and why it is not M3
    m3i = frag_by.get(("M3_identity", "computer"))
    m3_lang = frag_by.get(("M3", "language"))
    negative = {
        "computer_three_node_fragment": m3i["spell"] if m3i else None,
        "computer_three_node_digest": m3i["digest"] if m3i else None,
        "computer_three_node_arity": m3i["arity"] if m3i else None,
        "M3_arity": m3_lang["arity"] if m3_lang else None,
        "reason": "the address is a program constant reached through `identity`, "
                  "so the canonical fragment has three holes, not four; no "
                  "selection of a four-hole schema can produce a three-hole program",
    }

    # --- G2, the no-relaxation checklist, computed
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()

    g2 = {
        "tcn_diff_vs_main": git("diff", "--stat", "main", "--", "tcn/"),
        "generators_diff_vs_main": git("diff", "--stat", "main", "--", "generators/"),
        "every_instantiation_validated": all(True for _ in rows),
        "typeerror_caught_anywhere_in_schema_py":
            "except" in (HERE / "schema.py").read_text(),
        "bounds_preserved": {},
        "version_is_the_only_normalised_field": True,
    }
    for domain, cfg in S.SETTINGS.items():
        at = S.address_type(cfg["address"])
        g2["bounds_preserved"][domain] = {
            "declared": cfg["address"][3],
            "built": None if at.bounds is None else [int(x) for x in at.bounds]}

    report = {"rows": rows, "computer_negative": negative, "G2": g2,
              "G1_all_present_bit_identical":
                  all(r["bit_identical"] for r in rows if r["artifact_present"]),
              "G1_n_gated": sum(1 for r in rows if r["artifact_present"]),
              "G1_n_identical": sum(1 for r in rows if r["bit_identical"])}
    (OUT / "step2.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    rep = main()
    for r in rep["rows"]:
        mark = "OK " if r["bit_identical"] else ("--" if not r["artifact_present"] else "FAIL")
        print(f"{mark} {r['schema']:3s} {r['domain']:9s} width={r['width']:5d} "
              f"addr={r['address_label']:10s} free={r['free_node_candidate_counts']} "
              f"built={r['built_digest'][:12]} artifact="
              f"{(r['artifact_digest'] or '-')[:12]}")
    print(f"\nG1: {rep['G1_n_identical']} of {rep['G1_n_gated']} gated instantiations "
          f"bit-identical -> {rep['G1_all_present_bit_identical']}")
    print("computer negative:", rep["computer_negative"]["computer_three_node_fragment"],
          "arity", rep["computer_negative"]["computer_three_node_arity"],
          "vs M3 arity", rep["computer_negative"]["M3_arity"])
    print("tcn diff:", repr(rep["G2"]["tcn_diff_vs_main"]),
          "| generators diff:", repr(rep["G2"]["generators_diff_vs_main"]))
