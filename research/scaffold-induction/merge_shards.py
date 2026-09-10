"""Merge `cases_<domain>_s<i>.json` shards into `cases_<domain>.json`.

Shards partition the defect enumeration by index, so their case lists are
disjoint by construction; the merge concatenates them in shard order and sums
the rejection counts.  The base-scaffold decision is identical in every shard
and is asserted equal rather than taken from the first.

    python merge_shards.py <domain> [<domain> ...]
"""
from __future__ import annotations

import sys

import kit

SUM = ("defects_tried", "admitted", "rejected_solvable_on_train",
       "rejected_no_repair", "rejected_invalid", "seconds")


def merge(domain):
    paths = sorted(kit.OUT.glob(f"cases_{domain}_s*.json"))
    if not paths:
        print(f"{domain}: no shards")
        return
    parts = [kit.load(p.stem) for p in paths]
    out = dict(parts[0])
    out["cases"] = []
    out["shards"] = [p.name for p in paths]
    for k in SUM:
        out[k] = sum(p[k] for p in parts)
    out["peak_rss_gb"] = max(p["peak_rss_gb"] for p in parts)
    out["defects_tried"] = parts[0]["defects_tried"]      # the same full list per shard
    for p in parts:
        assert p["base"] == parts[0]["base"], f"{domain}: shards disagree on the base"
        assert p["n_train"] == parts[0]["n_train"]
        out["cases"] += p["cases"]
    print(f"{domain}: {len(paths)} shards, {out['admitted']} admitted, "
          f"{len(out['cases'])} case records")
    kit.dump(f"cases_{domain}", out)


if __name__ == "__main__":
    for d in sys.argv[1:] or ["arith"]:
        merge(d)
