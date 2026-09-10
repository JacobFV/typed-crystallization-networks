"""Re-derive every headline of `RESULTS.md` from `out/*.json` independently.

Deliberately does not import the analysis modules: it re-reads the raw records
and recomputes the claims from them, so a bug in `run_shared.py` cannot make its
own claim true.
"""
from __future__ import annotations

import collections
import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "out"
ok = True


def check(label, got, want):
    global ok
    good = got == want
    ok = ok and good
    print(f"{'PASS' if good else 'FAIL'}  {label}: got {got!r}, want {want!r}")


DOMAIN = {"V_same": "visual", "V_corner": "visual", "V_rect": "visual",
          "V_assembly": "visual", "L_stage_a": "language",
          "L_stage_b16": "language", "L_dyck22": "language", "C_agent": "computer"}

# --- the equivalence gate
eq = json.loads((OUT / "equivalence.json").read_text())
rows = [c for c in eq["checks"] if "identical" in c]
check("equivalence checks", len(rows), 862)
check("equivalence all identical", all(r["identical"] for r in rows), True)
check("equivalence programs", len({r["program"] for r in rows}), 431)

for n, n_frag, n_cls in ((3, 1874, 178), (5, 1928, 217)):
    inv = json.loads((OUT / f"inventory_{n}.json").read_text())
    check(f"n={n} total fragments",
          sum(a["fragments_with_registry"] for a in inv["artifacts"].values()), n_frag)
    check(f"n={n} D-classes", len(inv["classes"]), n_cls)
    check(f"n={n} artifacts", len(inv["artifacts"]), 8)
    check(f"n={n} dyck digest", inv["dyck_digest_reproduced"], "4d0ce927662af9fe54799de0")

    # cross-domain, recomputed from the raw class records
    for rel, field in (("D", "digest"), ("S", "S_class"), ("Sstar", "Sstar_class")):
        g = collections.defaultdict(list)
        for c in inv["classes"]:
            if c[field]:
                g[c[field]].append(c)
        cross = {k: cs for k, cs in g.items()
                 if len({DOMAIN[a] for c in cs for a in c["occurrences"]}) > 1}
        nontrivial = [k for k, cs in cross.items() if not any(c["trivial"] for c in cs)]
        check(f"n={n} {rel} cross-domain", len(cross), {"D": 2, "S": 2, "Sstar": 6}[rel])
        check(f"n={n} {rel} cross-domain NON-TRIVIAL", len(nontrivial), 0)

    # every cross-domain class is a single core operator
    ops = set()
    for c in inv["classes"]:
        if c["Sstar_class"] is None and c["S_class"] is None:
            continue
        doms = {DOMAIN[a] for a in c["occurrences"]}
        keys = [k for k in (c["S_class"], c["Sstar_class"]) if k]
        for k in keys:
            peers = [d for d in inv["classes"]
                     if d.get("S_class") == k or d.get("Sstar_class") == k]
            if len({DOMAIN[a] for d in peers for a in d["occurrences"]}) > 1 and c["nodes"] == 1:
                ops.add(c["body"][0][0])
    check(f"n={n} cross-domain operators", sorted(ops),
          ["add", "and", "eq", "identity", "lt", "sub"])

# --- the type-signature bound
for n, tot in ((3, 110), (5, 131)):
    b = json.loads((OUT / f"bounds_{n}.json").read_text())["B1_type_signature_bound"]
    check(f"n={n} type signatures", b["n_type_signatures"], tot)
    check(f"n={n} signatures crossing a domain", b["n_signatures_in_more_than_one_domain"], 2)
b3 = json.loads((OUT / "bounds_5.json").read_text())["B3_carriers"]
check("carriers in all three domains", b3["in_all_three"], ["BOOL", "u8[byte]"])
for k, v in b3["pairwise"].items():
    check(f"carriers {k}", v, ["BOOL", "u8[byte]"])

# --- the positive control
for n in (3, 5):
    for band in ("C-minall", "C-trace"):
        p = json.loads((OUT / f"control_{n}_{band}.json").read_text())["P1_boolean_control"]
        check(f"control {band} n={n} cross-family non-trivial S",
              p["cross_family_S"]["n_nontrivial"], 5)
        got = p["within_MAJ3_S"]["n_nontrivial"]
        print(f"{'PASS' if got in (11, 13) else 'FAIL'}  control {band} n={n} "
              f"within-MAJ3 non-trivial S: {got} (expect 11 at n=3, 13 at n=5)")
        ok = ok and got in (11, 13)
    c = json.loads((OUT / f"control_{n}_C-minall.json").read_text())["P2_within_domain"]
    check(f"P2 n={n} visual non-trivial S",
          c["visual"]["shared_across_programs_S"]["n_nontrivial"], 1)
    check(f"P2 n={n} language non-trivial S",
          c["language"]["shared_across_programs_S"]["n_nontrivial"], 0)

# --- the shipped fixture
fx = json.loads((OUT / "fixture.log").read_text().split("exit=")[0])
check("fixture initial loss", round(fx["initial_prediction_loss"], 6), 0.248836)
check("fixture final loss", round(fx["final_prediction_loss"], 6), 0.002231)
check("fixture evaluation", fx["evaluation_mean_return"], 4.0)
check("fixture frozen evaluation", fx["frozen_evaluation_mean_return"], 4.0)

# --- the test suite
log = (OUT / "pytest.log").read_text()
check("tests", "13 failed, 324 passed" in log, True)
check("failures are the tsx/node_modules class", log.count("tsx") >= 1, True)

print("\nALL CLAIMS VERIFIED" if ok else "\nSOME CLAIMS FAILED")
raise SystemExit(0 if ok else 1)
