"""Print the semantic-identity ranked table for one corpus."""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).parent / "out"


def body_str(body):
    return "; ".join(f"{n}={op}({','.join(a)})" for op, n, a in body)


def main():
    variant = sys.argv[1]
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    d = json.loads((OUT / f"semantic_{variant}.json").read_text())
    print(variant, "entries", d["entries"], "eligible", d["eligible"],
          "rank1_is_maj3", d["rank1_computes_maj3"], "best_maj3_rank", d["best_maj3_rank"])
    print("| rank | representative body | arity | circuits pooled | tasks | entries | sites | saving | per entry | maj3 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in d["ranked"][:top]:
        print(f"| {r['rank']} | `{body_str(r['body'])}` | {r['holes']} | "
              f"{r['circuits_pooled']} | {len(r['tasks'])} | {r['entries']} | "
              f"{r['occurrences']} | {r['saving_bits']:,} | {r['saving_per_entry']:.0f} | "
              f"{'YES' if r['computes_maj3'] else ''} |")


if __name__ == "__main__":
    main()
