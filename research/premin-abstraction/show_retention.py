"""Print the retention tables straight from `out/corpora.json`."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "out"


def main():
    d = json.loads((OUT / "corpora.json").read_text())
    for scope in ("retention_full", "retention_capped"):
        print(f"\n### {scope}\n")
        print("| task | band | method | exhausted | programs | MAJ3 | M | M' | has C-min prog |")
        print("|---|---|---|---|---|---|---|---|---|")
        for t in d["tasks"]:
            for label in ("min", "plus1"):
                b = t["bands"][label]
                r = b[scope]
                n = r["MAJ3"]["programs"]
                print(f"| `{t['task']}` | {label} (L={b['length']}) | {b['method']} | "
                      f"{b['exhausted']} | {n} | "
                      f"{r['MAJ3']['programs_retaining']} ({r['MAJ3']['fraction']:.3f}) | "
                      f"{r['M']['programs_retaining']} ({r['M']['fraction']:.3f}) | "
                      f"{r['Mprime']['programs_retaining']} ({r['Mprime']['fraction']:.3f}) | "
                      f"{b['contains_c_min_program'] if label == 'min' else '--'} |")
    print("\n### per ordered triple, MAJ3, full sets\n")
    for t in d["tasks"]:
        for label in ("min", "plus1"):
            b = t["bands"][label]
            print(t["task"], label, b["retention_full"]["MAJ3"]["per_ordered_triple"])
    print("\n### verification / cost\n")
    for t in d["tasks"]:
        for label in ("min", "plus1"):
            b = t["bands"][label]
            print(t["task"], label, "verified_in_tcn", b["verified_in_tcn"],
                  "capped", b["capped"], "dfs_expanded", b.get("dfs_expanded"),
                  "wall %.1f" % b["wall_seconds"])
    print("\ntotal corpus build wall %.1f s (6 tasks in parallel)" % d["wall_seconds"])


if __name__ == "__main__":
    main()
