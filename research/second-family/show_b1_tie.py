"""Every member of B1's top tie group, on every corpus of every band.

F3 asks whether the frequency baseline matches the breadth-weighted objective.
If it does, the next question is *why*, and the answer here is that B1's score
never separates its top group at all: what decides rank 1 is
`objectives.rank`'s tie-break `(-nodes, digest)`.  This lists the group so the
alternative the tie-break did **not** pick can be run as an arm.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"


def main():
    for band in ("C-trace", "C-minall"):
        d = json.loads((OUT / f"ranked_{band}.json").read_text())
        for cname, rec in d["corpora"].items():
            rows = rec["tables"]["B1"]["ranked"]
            top = rows[0]["score"]
            tie = [r for r in rows if r["score"] == top]
            print(f"--- {band} {cname}  top_score={top:g}")
            for r in tie:
                print(f"    nodes={r['nodes']} arity={r['arity']} "
                      f"occ={r['occurrences']:4d} window={r['is_window']} "
                      f"{r['digest']}")


if __name__ == "__main__":
    main()
