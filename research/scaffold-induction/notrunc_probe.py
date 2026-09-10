"""What the candidate-truncation bound costs, measured (PREREGISTRATION A11).

`MAX_NEW` truncated the candidate list an edit adds, in `legal_candidates` order.
For `bool` that deleted the repair: `WIDEN(y, xor)` needs the wiring `(n1, n2)`,
which `legal_candidates` emits at index 55 of 64, past the bound of 48.  Nine of
`bool`'s eleven admissible defects were recorded as having no repair for that
reason alone.

This measures what removing the bound costs in edit-space size and in undecided
edits, so the amendment is made on numbers rather than on principle alone.

    python notrunc_probe.py <domain>
"""
from __future__ import annotations

import sys

import kit
import domains
import edits as E
import run_domain as R
from tcn.search import space_size


def probe(domain, keys=("WIDEN|y|xor",)):
    d = domains.BUILDERS[domain]()
    base, r, sig = d["program"], d["registry"], d["signals"]
    allep = d["train"] + d["heldout"]
    defects = R.enumerate_defects(base, r, [nd.name for nd in base.nodes])
    out = {"domain": domain, "cases": []}
    for kind, site, arg in defects:
        f = R.APPLY[kind](base, r, site, arg)
        if f is None:
            continue
        tr = R.decide(f, d["train"], sig, r)
        if not (tr["decided"] and not tr["conforming"] and tr["exhausted"]):
            continue
        row = {"defect": f"{kind}:{site}:{arg}"}
        for label, mn, mnn in (("truncated", 48, 12), ("untruncated", 10**9, 10**9)):
            E.MAX_NEW, E.MAX_NEW_NODE = mn, mnn
            E.TRUNCATED.clear()
            es = E.enumerate_edits(f, r)
            sizes = sorted(space_size(p) for _, p in es)
            over = sum(1 for s in sizes if s > R.MAX_SPACE)
            row[label] = {"n_edits": len(es), "over_cap": over,
                          "median_space": sizes[len(sizes) // 2],
                          "max_space": sizes[-1],
                          "n_truncated_edits": len(E.TRUNCATED)}
            for e, p in es:
                if e.key in keys:
                    res = R.decide(p, allep, sig, r)
                    row[label][e.key] = {
                        "space": res["space"], "decided": res["decided"],
                        "is_repair": res["conforming"],
                        "certificate": res["certificate"]}
        out["cases"].append(row)
        print(f"  {row['defect']}", flush=True)
        for label in ("truncated", "untruncated"):
            v = row[label]
            print(f"     {label:12s} edits {v['n_edits']:4d} over-cap {v['over_cap']:3d} "
                  f"median {v['median_space']:>9,} max {v['max_space']:>12,} "
                  f"cut {v['n_truncated_edits']:3d} "
                  f"{ {k: v[k]['is_repair'] for k in keys if k in v} }", flush=True)
    E.MAX_NEW, E.MAX_NEW_NODE = 48, 12
    print(kit.dump(f"notrunc_{domain}", out))


if __name__ == "__main__":
    probe(sys.argv[1] if len(sys.argv) > 1 else "bool")
