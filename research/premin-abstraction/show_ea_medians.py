"""Re-derive §44's tight/wide medians from that track's raw records.

Reporting discipline: FINDINGS §44's headline table quotes a *median* accuracy
while `earned-abstraction/RESULTS.md` quotes the mean.  This recomputes both
from `earned-abstraction/out/gradient_*.json` so this track's table can be laid
beside §44's without taking either on trust.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

EA = Path(__file__).parent.parent / "earned-abstraction" / "out"

for scaffold in ("tight", "wide"):
    d = json.loads((EA / f"gradient_{scaffold}.json").read_text())
    print(f"--- {scaffold}")
    for s in d["summary"]:
        acc = [r["accuracy"] for r in d["records"] if r["arm"] == s["arm"]]
        print(f"{s['arm']:24s} solved {s['solved']}/{s['seeds']} "
              f"median {statistics.median(acc):.4f} mean {sum(acc)/len(acc):.4f} "
              f"best {max(acc):.4f}")
