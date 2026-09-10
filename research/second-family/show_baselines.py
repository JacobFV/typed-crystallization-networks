"""Constant and random baselines for every task, and the space each arm swept."""
from __future__ import annotations

import json
from pathlib import Path

import _paths  # noqa: F401

import evaltasks

OUT = Path(__file__).resolve().parent / "out"


def main():
    for t, fn in sorted(evaltasks.COMPACT_TASKS.items()):
        print(f"{t:20s} constant {evaltasks.constant_baseline(fn):.4f}  random 0.5000")
    print()
    d = json.loads((OUT / "heldout.json").read_text())
    seen = {}
    bad = []
    for r in d["rows"]:
        seen[(r["band"], r["spec"])] = (r["space_size"], r["candidates_per_node"])
        if not (r["exhausted"] and r["evaluated"] == r["space_size"]):
            bad.append(r)
    for k in sorted(seen):
        print(k, "space", seen[k][0], "candidates", seen[k][1])
    print("IC4 violations in heldout.json:", len(bad))
    a = json.loads((OUT / "arms_enum.json").read_text())
    bad2 = [r for r in a if not (r["exhausted"] and r["evaluated"] == r["space_size"])]
    print("IC4 violations in arms_enum.json:", len(bad2), "rows", len(a))


if __name__ == "__main__":
    main()
