"""Re-derive the certified minimum length of every earlier task, independently.

`corpora.min_lengths()` reads §44's figure.  This does not: it exhausts every
length from 0 upward with this track's own enumerator and records the count at
each length, so the minimum used to define the `min` and `plus1` bands rests on
a certificate produced here as well as on §44's.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import enumerate_programs as E
from corpus import EARLIER_TASKS

OUT = Path(__file__).parent / "out"
N = 4


def one(name):
    fn = dict(EARLIER_TASKS)[name]
    tt = E.table_of(N, fn)
    counts, expanded, t0 = [], [], time.perf_counter()
    for L in range(0, 7):
        sols, exhausted, exp = E.exhaustive(N, tt, L)
        counts.append({"length": L, "programs": len(sols), "exhausted": bool(exhausted),
                       "dfs_expanded": exp})
        expanded.append(exp)
        if len(sols) > 0:
            return {"task": name, "min_gates": L,
                    "certificate": f"lengths {list(range(L))} exhausted with 0 programs",
                    "by_length": counts, "total_expanded": sum(expanded),
                    "wall_seconds": time.perf_counter() - t0}
    return {"task": name, "min_gates": None, "by_length": counts,
            "total_expanded": sum(expanded), "wall_seconds": time.perf_counter() - t0}


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=6) as pool:
        recs = list(pool.map(one, [n for n, _ in EARLIER_TASKS]))
    for r in recs:
        print(r["task"], "min", r["min_gates"], r["certificate"],
              "counts", [c["programs"] for c in r["by_length"]],
              "expanded", r["total_expanded"], "wall %.1f" % r["wall_seconds"], flush=True)
    (OUT / "min_certificate.json").write_text(json.dumps(
        {"records": recs, "wall_seconds": time.perf_counter() - t0}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
