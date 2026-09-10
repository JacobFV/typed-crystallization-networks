"""Step 1: the fragment inventory and the three exact identity relations.

Writes `out/inventory_<max_nodes>.json`: one row per distinct canonical
fragment (D-class), with its S and S* identities where those are exactly
computable and the reason where they are not.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import frag
import identity as I
import artifacts as A

OUT = HERE / "out"


def body(canon):
    return [[n.candidates[0].operator.name, n.name, list(n.candidates[0].sources)]
            for n in canon.nodes]


def run(max_nodes, max_holes=frag.MAX_HOLES, cap=I.EXHAUST_CAP):
    arts, dyck_digest = A.all_artifacts()
    per_artifact = {}
    classes = {}          # digest -> record
    t_start = time.perf_counter()

    for name, (prog, reg) in arts.items():
        t0 = time.perf_counter()
        rows_reg = frag.fragments(prog, max_nodes, max_holes, registry=reg)
        rows_bare = frag.fragments(prog, max_nodes, max_holes, registry=None)
        per_artifact[name] = {
            "domain": A.DOMAIN[name],
            "nodes": len(prog.nodes),
            "program_digest": prog.digest,
            "fragments_with_registry": len(rows_reg),
            "fragments_mine_py_exact": len(rows_bare),
            "distinct_D_with_registry": len({c.digest for _, _, c, _ in rows_reg}),
            "distinct_D_mine_py_exact": len({c.digest for _, _, c, _ in rows_bare}),
            "seconds": time.perf_counter() - t0,
        }
        for root, S, canon, holes in rows_reg:
            rec = classes.setdefault(canon.digest, {
                "digest": canon.digest, "nodes": len(canon.nodes),
                "arity": len(canon.inputs), "body": body(canon),
                "input_types": [I.tag(t) for _, t in canon.inputs],
                "output_type": I.tag(canon.nodes[-1].output),
                "occurrences": collections.Counter(), "_canon": canon, "_reg": reg,
            })
            rec["occurrences"][name] += 1

    # identity, computed once per D-class
    for digest, rec in classes.items():
        canon, reg = rec.pop("_canon"), rec.pop("_reg")
        sig, why = I.signature(canon, reg, cap)
        rec["S_exhaustible"] = sig is not None
        rec["S_reason"] = why
        if sig is not None:
            key, _ = I.s_class(canon, reg, cap)
            rec["S_class"] = key
            rec["S_domain_rows"] = len(sig)
            triv, twhy = I.is_trivial(canon, sig)
        else:
            rec["S_class"] = None
            rec["S_domain_rows"] = None
            triv, twhy = I.is_trivial(canon, None)
        rec["trivial"] = triv
        rec["trivial_reason"] = twhy
        sstar, swhy = I.sstar_class(canon, widths=(4, 8), cap=cap)
        rec["Sstar_class"] = sstar
        rec["Sstar_reason"] = swhy
        rec["occurrences"] = dict(rec["occurrences"])
        rec["domains"] = sorted({A.DOMAIN[k] for k in rec["occurrences"]})

    report = {
        "max_nodes": max_nodes, "max_holes": max_holes, "exhaust_cap": cap,
        "dyck_digest_reproduced": dyck_digest,
        "artifacts": per_artifact,
        "n_D_classes": len(classes),
        "classes": sorted(classes.values(), key=lambda r: (-len(r["domains"]), r["digest"])),
        "seconds": time.perf_counter() - t_start,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"inventory_{max_nodes}.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-nodes", type=int, default=5)
    a = ap.parse_args()
    r = run(a.max_nodes)
    print(f"max_nodes={r['max_nodes']}  D-classes={r['n_D_classes']}  {r['seconds']:.1f}s")
    for k, v in r["artifacts"].items():
        print(f"  {k:12s} {v['domain']:9s} nodes={v['nodes']:4d} "
              f"frags={v['fragments_with_registry']:6d} "
              f"(mine.py-exact {v['fragments_mine_py_exact']:6d})  "
              f"D={v['distinct_D_with_registry']:5d}")
    multi = [c for c in r["classes"] if len(c["domains"]) > 1]
    print("cross-domain D-classes:", len(multi))
