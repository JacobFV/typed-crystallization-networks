"""Print one corpus variant's ranked table from `out/proposal_<variant>.json`."""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).parent / "out"


def body_str(body):
    return "; ".join(f"{n}={op}({','.join(a)})" for op, n, a in body)


def main():
    variant = sys.argv[1]
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    d = json.loads((OUT / f"proposal_{variant}.json").read_text())
    print(variant, "entries", d["entries"], "eligible", d["eligible"],
          "rank1_is_maj3", d["rank1_computes_maj3"], "best_maj3_rank", d["best_maj3_rank"],
          "mine wall %.1fs" % d["mine_wall_seconds"])
    print("| rank | body | arity | tasks | entries | sites | saving | per entry | maj3 |")
    print("|---|---|---|---|---|---|---|---|---|")
    rows = d["ranked"][:top] + [r for r in d["ranked"][top:] if r["computes_maj3"]]
    for r in rows:
        print(f"| {r['rank']} | `{body_str(r['body'])}` | {r['holes']} | {len(r['tasks'])} | "
              f"{r['entries']} | {r['occurrences']} | {r['saving_bits']} | "
              f"{r['saving_per_entry']:.0f} | {'YES' if r['computes_maj3'] else ''} |")


if __name__ == "__main__":
    main()
