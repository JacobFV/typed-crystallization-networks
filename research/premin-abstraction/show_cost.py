"""The compute cost of every corpus, beside §44's cost for its own."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "out"
EA = HERE.parent / "earned-abstraction" / "out"


def main():
    ea = json.loads((EA / "corpus_exact.json").read_text())
    ea_exp = sum(r["total_expanded"] for r in ea["report"] if r.get("solved"))
    ea_cpu = sum(r["wall_seconds"] for r in ea["report"])
    print(f"C-min (§44's own build): {ea_exp:,} DFS nodes, {ea_cpu:.1f} CPU-s, "
          f"{ea['wall_seconds']:.1f} s wall (6 tasks in parallel)")

    d = json.loads((OUT / "corpora.json").read_text())
    bands = {"min": [0, 0.0], "plus1": [0, 0.0]}
    for t in d["tasks"]:
        for label in bands:
            b = t["bands"][label]
            bands[label][0] += b.get("dfs_expanded", 0)
            bands[label][1] += b["wall_seconds"]
    for label, (exp, cpu) in bands.items():
        print(f"band {label:6s}: {exp:,} DFS nodes, {cpu:.1f} CPU-s")
    tot_exp = sum(v[0] for v in bands.values())
    tot_cpu = sum(v[1] for v in bands.values())
    print(f"C-trace total: {tot_exp:,} DFS nodes, {tot_cpu:.1f} CPU-s, "
          f"{d['wall_seconds']:.1f} s wall (6 tasks in parallel)")
    print(f"ratio to §44's corpus build: {tot_exp / ea_exp:.1f}x nodes, "
          f"{tot_cpu / ea_cpu:.1f}x CPU-s")

    cert = json.loads((OUT / "min_certificate.json").read_text())
    print("independent minimality certificate:",
          f"{sum(r['total_expanded'] for r in cert['records']):,} DFS nodes, "
          f"{sum(r['wall_seconds'] for r in cert['records']):.1f} CPU-s")

    for v in ("C-min", "C-minall", "C-plus1", "C-trace", "C-plus1-one"):
        p = OUT / f"proposal_{v}.json"
        if p.exists():
            j = json.loads(p.read_text())
            print(f"mine {v:12s}: {j['entries']:4d} entries, "
                  f"{j['mine_wall_seconds']:.1f} s")


if __name__ == "__main__":
    main()
