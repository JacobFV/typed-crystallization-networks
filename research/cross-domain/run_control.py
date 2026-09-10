"""The positive control the negative needs, and the within-domain comparison.

A "we found nothing non-trivial" result is worth nothing unless the same code,
unchanged, finds something non-trivial where something non-trivial is known to
be.  Two arms, both using `frag.py` and `identity.py` exactly as the inventory
does:

* **P1, the Boolean positive control.**  §52's MAJ3 corpus and §57's W4 corpus,
  treated as two pseudo-domains.  §52 established that a non-trivial semantic
  class (the 3-ary window function) is shared across the MAJ3 tasks, so the
  machinery must find non-trivial shared classes *within* MAJ3.  Whether it also
  finds them *across* the two Boolean families is the quantity that makes the
  real-domain zero interpretable: same identity relation, same rule, Boolean
  carriers on both sides.

* **P2, within-domain sharing in the real artifacts.**  For each real domain,
  classes occurring in two or more of that domain's own frozen programs.  This
  separates "the machinery finds nothing" from "the machinery finds plenty, but
  never across a domain boundary".
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    sys.path.insert(0, p)

import frag
import identity as I
import artifacts as A
from run_shared import spell

OUT = HERE / "out"
CAP_TASKS = 8   # tasks per Boolean band; §52's corpora are larger than we need


def _bool_corpora(band):
    sys.path.insert(0, str(ROOT / "research" / "semantic-library"))
    import family as fam1
    corpus1, _ = fam1.corpus_for("maj", band, ())
    sys.path.insert(0, str(ROOT / "research" / "second-family"))
    sys.modules.pop("family", None)
    import family as fam2
    corpus2, _ = fam2.corpus_for("w4", band, ())
    return corpus1, corpus2


def _classes(progs, domain_of, max_nodes, cap=I.EXHAUST_CAP):
    """digest -> record, exactly as `run_inventory` builds it."""
    classes = {}
    for name, (prog, reg) in progs.items():
        for root, S, canon, holes in frag.fragments(prog, max_nodes, frag.MAX_HOLES, registry=reg):
            rec = classes.setdefault(canon.digest, {
                "digest": canon.digest, "nodes": len(canon.nodes),
                "arity": len(canon.inputs),
                "body": [[n.candidates[0].operator.name, n.name, list(n.candidates[0].sources)]
                         for n in canon.nodes],
                "occurrences": collections.Counter(), "_canon": canon, "_reg": reg})
            rec["occurrences"][name] += 1
    for rec in classes.values():
        canon, reg = rec.pop("_canon"), rec.pop("_reg")
        sig, why = I.signature(canon, reg, cap)
        rec["S_class"], _ = (I.s_class(canon, reg, cap) if sig is not None else (None, why))
        rec["S_reason"] = why
        triv, twhy = I.is_trivial(canon, sig)
        rec["trivial"], rec["trivial_reason"] = triv, twhy
        rec["occurrences"] = dict(rec["occurrences"])
        rec["domains"] = sorted({domain_of[k] for k in rec["occurrences"]})
    return classes


def _shared(classes, field):
    groups = collections.defaultdict(list)
    for c in classes.values():
        k = c["digest"] if field == "D" else c[field]
        if k is None:
            continue
        groups[k].append(c)
    rows = []
    for k, cs in groups.items():
        doms = sorted({d for c in cs for d in c["domains"]})
        if len(doms) < 2:
            continue
        # Amendment A2: `any`, not `all`
        rows.append({"key": k, "domains": doms, "trivial": any(c["trivial"] for c in cs),
                     "nodes": min(c["nodes"] for c in cs),
                     "spell": spell(min(cs, key=lambda c: c["nodes"])),
                     "members": [{"digest": c["digest"], "nodes": c["nodes"],
                                  "occurrences": c["occurrences"]} for c in cs]})
    rows.sort(key=lambda r: (r["trivial"], -r["nodes"]))
    return rows


def p1(max_nodes, band="C-minall"):
    c1, c2 = _bool_corpora(band)
    names1 = sorted(c1)[:CAP_TASKS]
    names2 = sorted(c2)[:CAP_TASKS]
    progs, dom = {}, {}
    for n in names1:
        progs[f"maj/{n}"] = (c1[n], None)
        dom[f"maj/{n}"] = "boolMAJ3"
    for n in names2:
        progs[f"w4/{n}"] = (c2[n], None)
        dom[f"w4/{n}"] = "boolW4"
    t0 = time.perf_counter()
    classes = _classes(progs, dom, max_nodes)
    rows_d, rows_s = _shared(classes, "D"), _shared(classes, "S_class")
    # within-MAJ3 sharing: same machinery, pseudo-domain = task
    dom_task = {k: k for k in progs if k.startswith("maj/")}
    inner = _classes({k: v for k, v in progs.items() if k.startswith("maj/")}, dom_task, max_nodes)
    rows_inner = _shared(inner, "S_class")
    return {
        "band": band, "max_nodes": max_nodes,
        "tasks": {"boolMAJ3": names1, "boolW4": names2},
        "n_D_classes": len(classes),
        "cross_family_D": {"n": len(rows_d),
                           "n_nontrivial": sum(1 for r in rows_d if not r["trivial"]),
                           "rows": rows_d[:40]},
        "cross_family_S": {"n": len(rows_s),
                           "n_nontrivial": sum(1 for r in rows_s if not r["trivial"]),
                           "rows": rows_s[:40]},
        "within_MAJ3_S": {"n": len(rows_inner),
                          "n_nontrivial": sum(1 for r in rows_inner if not r["trivial"]),
                          "rows": rows_inner[:40]},
        "seconds": time.perf_counter() - t0,
    }


def p2(max_nodes):
    arts, _ = A.all_artifacts()
    out = {}
    for domain in ("visual", "language", "computer"):
        progs = {k: v for k, v in arts.items() if A.DOMAIN[k] == domain}
        if len(progs) < 2:
            out[domain] = {"n_programs": len(progs), "note": "one program only"}
            continue
        dom_prog = {k: k for k in progs}
        classes = _classes(progs, dom_prog, max_nodes)
        rows = _shared(classes, "S_class")
        rows_d = _shared(classes, "D")
        out[domain] = {
            "n_programs": len(progs), "n_D_classes": len(classes),
            "shared_across_programs_D": {"n": len(rows_d),
                                         "n_nontrivial": sum(1 for r in rows_d if not r["trivial"]),
                                         "rows": rows_d[:25]},
            "shared_across_programs_S": {"n": len(rows), "n_nontrivial":
                                         sum(1 for r in rows if not r["trivial"]),
                                         "rows": rows[:25]},
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-nodes", type=int, default=3)
    ap.add_argument("--band", default="C-minall")
    a = ap.parse_args()
    rep = {"P1_boolean_control": p1(a.max_nodes, a.band), "P2_within_domain": p2(a.max_nodes)}
    OUT.mkdir(exist_ok=True)
    (OUT / f"control_{a.max_nodes}_{a.band}.json").write_text(json.dumps(rep, indent=1))
    P = rep["P1_boolean_control"]
    print(f"P1 band={P['band']} max_nodes={P['max_nodes']}  D-classes={P['n_D_classes']}")
    print(f"   within MAJ3, S:        {P['within_MAJ3_S']['n']:4d} shared, "
          f"{P['within_MAJ3_S']['n_nontrivial']:4d} NON-TRIVIAL")
    print(f"   MAJ3 vs W4, D:         {P['cross_family_D']['n']:4d} shared, "
          f"{P['cross_family_D']['n_nontrivial']:4d} NON-TRIVIAL")
    print(f"   MAJ3 vs W4, S:         {P['cross_family_S']['n']:4d} shared, "
          f"{P['cross_family_S']['n_nontrivial']:4d} NON-TRIVIAL")
    for r in P["cross_family_S"]["rows"][:6]:
        print(f"        {'TRIV' if r['trivial'] else 'NON-TRIVIAL':11s} {r['nodes']}n {r['spell'][:70]}")
    print("P2 within-domain sharing across a domain's own frozen programs:")
    for d, v in rep["P2_within_domain"].items():
        if "note" in v:
            print(f"   {d:9s} {v['note']}")
            continue
        print(f"   {d:9s} D {v['shared_across_programs_D']['n']:4d} shared "
              f"({v['shared_across_programs_D']['n_nontrivial']} non-trivial)   "
              f"S {v['shared_across_programs_S']['n']:4d} shared "
              f"({v['shared_across_programs_S']['n_nontrivial']} non-trivial)")
        for r in v["shared_across_programs_S"]["rows"][:4]:
            print(f"        {'TRIV' if r['trivial'] else 'NON-TRIVIAL':11s} {r['nodes']}n {r['spell'][:70]}")
