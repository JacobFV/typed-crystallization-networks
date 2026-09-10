"""Mining cost and identity fragmentation, per band -- §52's two side measurements.

§52 reported that pooling costs 1.08-1.19x syntactic mining and *shrinks* the
eligible set, and that `MAJ3` was realised by 8 to 12 distinct digests under
syntactic identity.  Both are re-measured here on the second family from the
ranked tables already on disk.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"


def main():
    for band in ("C-trace", "C-minall"):
        sem = json.loads((OUT / f"mined_w4_{band}_semantic.json").read_text())
        syn = json.loads((OUT / f"mined_w4_{band}_syntactic.json").read_text())
        winner = [r for r in syn["ranked"] if r["is_window"]]
        print(f"{band}: entries {sem['entries']}, eligible semantic "
              f"{sem['eligible']} vs syntactic {syn['eligible']}, "
              f"pooling cost {sem['mine_wall_seconds'] / syn['mine_wall_seconds']:.2f}x "
              f"({sem['mine_wall_seconds']:.2f}s vs {syn['mine_wall_seconds']:.2f}s)")
        print(f"    window: {len(winner)} distinct digests under syntactic identity, "
              f"best syntactic rank {syn['best_window_rank']}, "
              f"pooled rank {sem['best_window_rank']}, "
              f"circuits pooled into the class "
              f"{[r['circuits_pooled'] for r in sem['ranked'] if r['is_window']]}")


if __name__ == "__main__":
    main()
