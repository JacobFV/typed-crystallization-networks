"""How the majority's mass is split across canonical realisations, per corpus.

A frequency rule ranks *digests*, not functions.  This prints, per corpus, how
many distinct canonical abstractions compute majority, how much corpus mass
each one carries, and what the single largest one carries -- beside the rank-1
proposal, which is what it has to beat.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "out"
VARIANTS = ("C-min", "C-minall", "C-plus1", "C-trace", "C-plus1-one")


def main():
    print("| corpus | entries | eligible | rank-1 arity/nodes | rank-1 tasks/entries/saving | "
          "maj3 digests | maj3 entries (summed) | best maj3 digest: tasks/entries/saving | "
          "best maj3 rank |")
    print("|---|---|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        p = OUT / f"proposal_{v}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        top = d["ranked"][0] if d["ranked"] else None
        maj = d["maj3_proposals"]
        best = min(maj, key=lambda r: r["rank"]) if maj else None
        print(f"| `{v}` | {d['entries']} | {d['eligible']} | "
              f"{top['holes']}/{top['nodes']} | "
              f"{len(top['tasks'])}/{top['entries']}/{top['saving_bits']:,} | "
              f"{len(maj)} | {sum(r['entries'] for r in maj)} | "
              f"{(str(len(best['tasks'])) + '/' + str(best['entries']) + '/' + format(best['saving_bits'], ',')) if best else '--'} | "
              f"{d['best_maj3_rank']} |")


if __name__ == "__main__":
    main()
