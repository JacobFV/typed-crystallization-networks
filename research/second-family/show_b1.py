"""How B1's rank 1 is actually decided -- falsification F3, checked hardest.

`B1` scores a class by the number of distinct corpus tasks it occurs in.  If
that score never separates the window from its competitors, then B1's apparent
win is settled entirely by `objectives.rank`'s tie-break `(-nodes, digest)`
and not by frequency at all.  This prints the tie group at the top score on
every corpus of every band.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"


def main():
    for band in ("C-trace", "C-minall"):
        d = json.loads((OUT / f"ranked_{band}.json").read_text())
        print("=====", band)
        for cname, rec in d["corpora"].items():
            rows = rec["tables"]["B1"]["ranked"]
            top = rows[0]["score"]
            tie = [r for r in rows if r["score"] == top]
            maxn = max(r["nodes"] for r in tie)
            atmax = [r for r in tie if r["nodes"] == maxn]
            win = next((r for r in rows if r["is_window"]), None)
            print(f"{cname:8s} B1 top_score={top:g} tied={len(tie):3d} "
                  f"node_counts={sorted({r['nodes'] for r in tie})} "
                  f"at_max_nodes={len(atmax)} "
                  f"window_in_tie={any(r['is_window'] for r in tie)} "
                  f"window_nodes={win['nodes'] if win else None} "
                  f"decided_by={rec['tables']['B1']['rank1_decided_by']}")
            rows2 = rec["tables"]["B2"]["ranked"]
            w2 = next((r for r in rows2 if r["is_window"]), None)
            print(f"{'':8s} B2 rank1="
                  f"{'WIN' if rows2[0]['is_window'] else rows2[0]['digest'][:8]} "
                  f"window_rank={w2['rank'] if w2 else None} "
                  f"window_occ={w2['occurrences'] if w2 else None} "
                  f"top_occ={rows2[0]['occurrences']}")


if __name__ == "__main__":
    main()
