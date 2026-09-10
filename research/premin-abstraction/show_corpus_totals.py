"""Corpus-level retention totals, for the capped corpora that were actually mined."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "out"


def main():
    d = json.loads((OUT / "corpora.json").read_text())
    for scope, label in (("retention_capped", "mined (capped)"),
                         ("retention_full", "full enumeration")):
        print(f"\n{label}")
        for bands, name in ((("min",), "C-minall"), (("plus1",), "C-plus1"),
                            (("min", "plus1"), "C-trace")):
            tot = {"n": 0, "MAJ3": 0, "M": 0, "Mprime": 0}
            for t in d["tasks"]:
                for b in bands:
                    r = t["bands"][b][scope]
                    tot["n"] += r["MAJ3"]["programs"]
                    for k in ("MAJ3", "M", "Mprime"):
                        tot[k] += r[k]["programs_retaining"]
            print(f"  {name:10s} n={tot['n']:5d} "
                  f"MAJ3 {tot['MAJ3']:5d} ({tot['MAJ3']/tot['n']:.3f})  "
                  f"M {tot['M']:5d} ({tot['M']/tot['n']:.3f})  "
                  f"M' {tot['Mprime']:5d} ({tot['Mprime']/tot['n']:.3f})")


if __name__ == "__main__":
    main()
