"""Step 1 headline: which identity classes span more than one domain, and are any non-trivial.

Reads `out/inventory_<n>.json` and resolves the pre-registered questions.  No
ranking, no `description_bits`: this counts classes, it does not score them.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"


def spell(rec):
    """A readable form of the canonical fragment: nested operator applications."""
    src = {}
    for op, name, ss in rec["body"]:
        src[name] = (op, ss)

    def show(k, d=0):
        if k not in src or d > 6:
            return k
        op, ss = src[k]
        return f"{op}(" + ", ".join(show(s, d + 1) for s in ss) + ")"

    return show(rec["body"][-1][1])


def tshort(tag):
    """Display half of the `label#sha` tag written by `run_inventory`."""
    return tag.split("#")[0]


def report(max_nodes):
    inv = json.loads((OUT / f"inventory_{max_nodes}.json").read_text())
    classes = inv["classes"]
    lines = []
    out = {"max_nodes": max_nodes, "n_D_classes": len(classes)}

    # --- how far exactness reaches
    ex = collections.Counter()
    reasons = collections.Counter()
    for c in classes:
        ex["S exact" if c["S_exhaustible"] else "S not exhaustible"] += 1
        if not c["S_exhaustible"]:
            reasons[c["S_reason"]] += 1
        ex["S* exact" if c["Sstar_class"] else "S* not computable"] += 1
    out["exactness"] = dict(ex)
    out["S_unexhaustible_reasons"] = dict(reasons)

    # --- cross-domain, by each relation
    def cross(keyfield):
        groups = collections.defaultdict(list)
        for c in classes:
            k = c[keyfield] if keyfield != "D" else c["digest"]
            if k is None:
                continue
            groups[k].append(c)
        res = []
        for k, cs in groups.items():
            doms = sorted({d for c in cs for d in c["domains"]})
            if len(doms) > 1:
                res.append((k, doms, cs))
        return res

    for rel in ("D", "S_class", "Sstar_class"):
        rows = cross(rel)
        key = {"D": "D", "S_class": "S", "Sstar_class": "Sstar"}[rel]
        out[f"cross_domain_{key}"] = []
        for k, doms, cs in sorted(rows, key=lambda r: -len(r[1])):
            # Amendment A2: a class that can be spelled with one primitive IS
            # that primitive.  `any`, not `all` -- the conservative direction.
            triv = any(c["trivial"] for c in cs)
            out[f"cross_domain_{key}"].append({
                "key": k, "domains": doms,
                "members": [{"digest": c["digest"], "nodes": c["nodes"],
                             "arity": c["arity"], "spell": spell(c),
                             "in_types": [tshort(t) for t in c["input_types"]],
                             "out_type": tshort(c["output_type"]),
                             "occurrences": c["occurrences"],
                             "trivial": c["trivial"],
                             "trivial_reason": c["trivial_reason"]} for c in cs],
                "all_trivial": triv,
            })
        n_nontrivial = sum(0 if r["all_trivial"] else 1 for r in out[f"cross_domain_{key}"])
        out[f"n_cross_domain_{key}"] = len(rows)
        out[f"n_cross_domain_{key}_nontrivial"] = n_nontrivial
        lines.append(f"{key}: {len(rows)} cross-domain classes, {n_nontrivial} non-trivial")

    # --- three-domain classes
    for key in ("D", "S", "Sstar"):
        out[f"n_all_three_{key}"] = sum(
            1 for r in out[f"cross_domain_{key}"] if len(r["domains"]) == 3)

    (OUT / f"shared_{max_nodes}.json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-nodes", type=int, default=5)
    a = ap.parse_args()
    o = report(a.max_nodes)
    for key in ("D", "S", "Sstar"):
        print(f"\n=== {key}: {o['n_cross_domain_' + key]} cross-domain "
              f"({o['n_all_three_' + key]} in all three) ===")
        for r in o["cross_domain_" + key]:
            m = r["members"][0]
            print(f"  {'+'.join(r['domains']):24s} {'TRIV' if r['all_trivial'] else 'NON-TRIVIAL':11s} "
                  f"{m['nodes']}n a{m['arity']} {m['spell'][:70]}  "
                  f"{m['in_types']}->{m['out_type']}")
            for mm in r["members"]:
                print(f"        {mm['digest'][:12]} {mm['occurrences']}")
