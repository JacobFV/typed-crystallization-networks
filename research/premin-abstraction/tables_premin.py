"""Render the reported tables straight from the artifacts in out/."""
from __future__ import annotations

import json
from pathlib import Path

import arms_premin as arms

OUT = Path(__file__).parent / "out"


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def enum_table(scaffold="tight"):
    print("| arm | candidates/node | search space | evaluated | exhausted | conforming | "
          "certificate | live nodes | wall (s) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm in arms.ARMS:
        d = load(f"enum_{scaffold}_{arm}.json")
        if not d:
            print(f"| `{arm}` | (not run) | | | | | | | |")
            continue
        print(f"| `{arm}` | {'/'.join(str(c) for c in d['candidates_per_node'])} | "
              f"{d['space_size']:,} | {d['evaluated']:,} | {d['exhausted']} | "
              f"{d['conforming']} | `{d['certificate']}` | "
              f"{d.get('live_nodes', '--')} | {d['wall_seconds']:.0f} |")


def gradient_table(scaffold):
    d = load(f"gradient_{scaffold}.json")
    if not d:
        print(f"({scaffold} gradient not run)")
        return
    print("| arm | seeds solved | median steps | median accuracy | mean accuracy | best | "
          "constant baseline | module on output path | space |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in d["summary"]:
        print(f"| `{s['arm']}` | {s['solved']}/{s['seeds']} | "
              f"{s['median_steps_to_conformant'] if s['median_steps_to_conformant'] is not None else '--'} | "
              f"{s['median_accuracy']:.4f} | {s['mean_accuracy']:.4f} | "
              f"{s['best_accuracy']:.4f} | 0.5000 | {s['module_on_output_path']} | "
              f"{s['space_size']:,} |")


def module_table():
    pub = load("published.json") or {}
    print("| label | corpus | pick | rank | tasks | entries | sites | saving (bits) | "
          "operator | nodes | maj3 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for label, r in sorted(pub.items()):
        print(f"| `{label}` | {r['variant']} | {r['pick']} | {r['rank']} | "
              f"{len(r['tasks'])} | {r['entries']} | {r['occurrences']} | "
              f"{r['saving_bits']:,} | `{r['operator']}` | {r['nodes']} | "
              f"{'YES' if r['computes_maj3'] else 'no'} |")


if __name__ == "__main__":
    print("### modules published\n")
    module_table()
    print("\n### tight scaffold, exhaustive enumeration\n")
    enum_table("tight")
    print("\n### tight scaffold, gradient\n")
    gradient_table("tight")
    print("\n### wide scaffold, gradient\n")
    gradient_table("wide")
