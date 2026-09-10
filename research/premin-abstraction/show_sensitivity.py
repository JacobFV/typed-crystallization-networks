"""Print the parameter sweep from `out/sensitivity.json`."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "out"


def body_str(body):
    return "; ".join(f"{n}={op}({','.join(a)})" for op, n, a in body) if body else "--"


def main():
    d = json.loads((OUT / "sensitivity.json").read_text())
    for variant in sorted({r["variant"] for r in d["records"]}):
        rows = [r for r in d["records"] if r["variant"] == variant]
        print(f"\n### {variant}\n")
        seen = {}
        for r in rows:
            seen.setdefault(r["rank1_digest"], []).append(r)
        for digest, group in sorted(seen.items(), key=lambda kv: -len(kv[1])):
            g = group[0]
            holes = sorted({x["max_holes"] for x in group})
            nodes = sorted({x["max_nodes"] for x in group})
            mt = sorted({x["min_tasks"] for x in group})
            print(f"* `{(digest or 'none')[:12]}` in **{len(group)} of {len(rows)}** settings "
                  f"(max_holes {holes}, max_nodes {nodes}, min_tasks {mt}): "
                  f"{g['rank1_nodes']} nodes, arity {g['rank1_holes']}, "
                  f"saving {g['rank1_saving_bits']}, maj3 {g['rank1_computes_maj3']} "
                  f"-- `{body_str(g['rank1_body'])}`")
        print("  settings whose rank-1 computes maj3:",
              sum(1 for r in rows if r["rank1_computes_maj3"]), "of", len(rows))
        ranks = {}
        for r in rows:
            ranks.setdefault(r["best_maj3_rank"], 0)
            ranks[r["best_maj3_rank"]] += 1
        print("  best maj3 rank ->", dict(sorted(ranks.items(),
                                                 key=lambda kv: (kv[0] is None, kv[0]))))


if __name__ == "__main__":
    main()
