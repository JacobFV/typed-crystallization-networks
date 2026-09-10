"""Which §57 regime are the real artifacts in?

§57: "pooling buys rank in proportion to fragmentation, and inverts where the
competitor is itself the fragmented class."  This track does no ranking, so
§57's inversion cannot arise here -- but the *fragmentation* it turns on is a
property of the artifacts and is worth stating.  Fragmentation is measured as
the collapse from D-classes (digest identity) to S-classes (semantic identity)
among the fragments whose carriers are exhaustible: a ratio of 1.00 means digest
identity already sees everything semantic identity sees, and pooling buys
nothing.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"


def run(max_nodes):
    inv = json.loads((OUT / f"inventory_{max_nodes}.json").read_text())
    per = collections.defaultdict(lambda: {"D": set(), "S": set(), "unexh_D": 0})
    for c in inv["classes"]:
        for art in c["occurrences"]:
            per[art]["D"].add(c["digest"])
            if c["S_class"]:
                per[art]["S"].add(c["S_class"])
            else:
                per[art]["unexh_D"] += 1
    rows = {}
    for art, v in per.items():
        exh_d = len(v["D"]) - v["unexh_D"]
        rows[art] = {"D_classes": len(v["D"]),
                     "D_classes_exhaustible": exh_d,
                     "S_classes": len(v["S"]),
                     "collapse_ratio": round(exh_d / len(v["S"]), 3) if v["S"] else None}
    rep = {"max_nodes": max_nodes, "per_artifact": rows}
    (OUT / f"regime_{max_nodes}.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-nodes", type=int, default=5)
    a = ap.parse_args()
    r = run(a.max_nodes)
    print(f"max_nodes={r['max_nodes']}")
    print(f"{'artifact':13s} {'D':>5s} {'D exh':>6s} {'S':>4s} {'D/S':>6s}")
    for k, v in r["per_artifact"].items():
        print(f"{k:13s} {v['D_classes']:5d} {v['D_classes_exhaustible']:6d} "
              f"{v['S_classes']:4d} {str(v['collapse_ratio']):>6s}")
