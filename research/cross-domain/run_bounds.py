"""Two exact bounds that do not depend on the exhaustion cap.

**B1 - the type-signature bound.**  Relation S is `(input types, output type,
exhaustive map)`.  Two fragments in different domains can be S-equal only if
their *type signatures* are equal, and the type signature is exact and free to
compute for every fragment, exhaustible or not.  So the number of type
signatures occurring in more than one domain is an **upper bound** on the number
of cross-domain S classes at any cap.  If a signature never co-occurs, no
exhaustion budget could ever produce a match there.  This is what makes the
`EXHAUST_CAP` honest rather than convenient.

**B2 - the operator-shape relation T.**  The canonical fragment with every type
erased: operator names and the edge structure only.  T is *not* a semantic
relation and nothing here claims it is; it answers the separate question
"is there any recurring multi-node motif across the domains at all, even one the
type system forbids identifying?".  A T match is reported as a structural
coincidence, never as a shared abstraction.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"

import sys
sys.path.insert(0, str(HERE))
from run_shared import spell, tshort  # noqa: E402


def typesig(c):
    """Exact: the tags carry a sha256 of the full `Type.to_dict()`."""
    return json.dumps({"in": c["input_types"], "out": c["output_type"]},
                      sort_keys=True, separators=(",", ":"))


def shape(c):
    """Operator names + edges, types erased."""
    payload = [[op, name, srcs] for op, name, srcs in c["body"]]
    return hashlib.sha256(json.dumps([payload, c["arity"]], sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()[:16]


def run(max_nodes):
    inv = json.loads((OUT / f"inventory_{max_nodes}.json").read_text())
    classes = inv["classes"]

    # ---- B1
    by_sig = collections.defaultdict(set)
    sig_classes = collections.defaultdict(list)
    for c in classes:
        s = typesig(c)
        by_sig[s] |= set(c["domains"])
        sig_classes[s].append(c)
    shared_sigs = {s: sorted(d) for s, d in by_sig.items() if len(d) > 1}
    b1 = {
        "n_type_signatures": len(by_sig),
        "n_signatures_in_more_than_one_domain": len(shared_sigs),
        "signatures": [],
    }
    for s, doms in sorted(shared_sigs.items(), key=lambda kv: -len(kv[1])):
        d = json.loads(s)
        b1["signatures"].append({
            "domains": doms,
            "in": [tshort(t) for t in d["in"]],
            "out": tshort(d["out"]),
            "n_D_classes": len(sig_classes[s]),
            "max_nodes_of_a_member": max(c["nodes"] for c in sig_classes[s]),
            "examples": sorted({spell(c) for c in sig_classes[s]})[:8],
        })

    # ---- B2
    by_shape = collections.defaultdict(list)
    for c in classes:
        by_shape[shape(c)].append(c)
    b2 = {"n_shapes": len(by_shape), "cross_domain": []}
    for k, cs in by_shape.items():
        doms = sorted({d for c in cs for d in c["domains"]})
        if len(doms) < 2:
            continue
        b2["cross_domain"].append({
            "shape": k, "domains": doms, "nodes": cs[0]["nodes"],
            "arity": cs[0]["arity"], "spell": spell(cs[0]),
            "members": [{"digest": c["digest"],
                         "in": [tshort(t) for t in c["input_types"]],
                         "out": tshort(c["output_type"]),
                         "occurrences": c["occurrences"]} for c in cs],
        })
    b2["cross_domain"].sort(key=lambda r: (-r["nodes"], -len(r["domains"])))
    b2["n_cross_domain"] = len(b2["cross_domain"])
    b2["n_cross_domain_multinode"] = sum(1 for r in b2["cross_domain"] if r["nodes"] > 1)
    b2["n_cross_domain_all_three"] = sum(1 for r in b2["cross_domain"] if len(r["domains"]) == 3)

    # ---- B3: the carrier inventory.  An S match needs identical hole types, so
    # the intersection of the domains' carrier sets bounds where any cross-domain
    # S class can live at all.  Exact, and independent of the exhaustion cap.
    carriers = collections.defaultdict(set)
    for c in classes:
        for t in list(c["input_types"]) + [c["output_type"]]:
            for d in c["domains"]:
                carriers[d].add(t)
    doms = sorted(carriers)
    inter3 = set.intersection(*[carriers[d] for d in doms]) if doms else set()
    b3 = {"per_domain": {d: sorted(tshort(t) for t in carriers[d]) for d in doms},
          "in_all_three": sorted(tshort(t) for t in inter3),
          "pairwise": {f"{a}+{b}": sorted(tshort(t) for t in carriers[a] & carriers[b])
                       for i, a in enumerate(doms) for b in doms[i + 1:]}}

    rep = {"max_nodes": max_nodes, "B1_type_signature_bound": b1,
           "B2_operator_shape": b2, "B3_carriers": b3}
    (OUT / f"bounds_{max_nodes}.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-nodes", type=int, default=5)
    a = ap.parse_args()
    r = run(a.max_nodes)
    b1 = r["B1_type_signature_bound"]
    print(f"B1: {b1['n_type_signatures']} distinct type signatures, "
          f"{b1['n_signatures_in_more_than_one_domain']} occur in >1 domain")
    for s in b1["signatures"]:
        print(f"   {'+'.join(s['domains']):24s} {s['in']}->{s['out']} "
              f"({s['n_D_classes']} D-classes, largest {s['max_nodes_of_a_member']} nodes)")
        for e in s["examples"]:
            print(f"        {e[:100]}")
    b3 = r["B3_carriers"]
    print("\nB3 carriers touched by each domain's fragments:")
    for d, ts in b3["per_domain"].items():
        print(f"   {d:9s} {ts}")
    print(f"   in all three: {b3['in_all_three']}")
    for k, v in b3["pairwise"].items():
        print(f"   {k:20s} {v}")
    b2 = r["B2_operator_shape"]
    print(f"\nB2: {b2['n_shapes']} operator shapes, {b2['n_cross_domain']} cross-domain, "
          f"{b2['n_cross_domain_multinode']} of them multi-node, "
          f"{b2['n_cross_domain_all_three']} in all three domains")
    for row in b2["cross_domain"]:
        print(f"   {row['nodes']}n a{row['arity']} {'+'.join(row['domains']):24s} {row['spell'][:80]}")
        for m in row["members"]:
            print(f"        {m['digest'][:12]} {m['in']}->{m['out']} {m['occurrences']}")
