"""Render the reported tables straight from the artifacts in out/."""
from __future__ import annotations

import json
from pathlib import Path

import arms

OUT = Path(__file__).parent / "out"


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def enum_table():
    print("| arm | candidates/node | search space | evaluated | exhausted | conforming | certificate | wall (s) |")
    print("|---|---|---|---|---|---|---|---|")
    for arm in arms.ARMS:
        d = load(f"enum_tight_{arm}.json")
        if not d:
            print(f"| {arm} | (not run) |")
            continue
        print(f"| `{arm}` | {'/'.join(str(c) for c in d['candidates_per_node'])} | "
              f"{d['space_size']:,} | {d['evaluated']:,} | {d['exhausted']} | "
              f"{d['conforming']} | `{d['certificate']}` | {d['wall_seconds']:.0f} |")


def gradient_table(scaffold):
    d = load(f"gradient_{scaffold}.json")
    if not d:
        print(f"({scaffold} gradient not run)")
        return
    print(f"| arm | seeds solved | median steps | mean steps | mean accuracy | "
          f"module on output path | s/step | space |")
    print("|---|---|---|---|---|---|---|---|")
    for s in d["summary"]:
        print(f"| `{s['arm']}` | {s['solved']}/{s['seeds']} | "
              f"{s['median_steps_to_conformant']} | "
              f"{('%.0f' % s['mean_steps_to_conformant']) if s['mean_steps_to_conformant'] is not None else '--'} | "
              f"{s['mean_accuracy']:.4f} | {s['module_on_output_path']} | "
              f"{s['mean_seconds_per_step']:.2f} | {s['space_size']:.3g} |")


def economy_table():
    for row in load("economy.json") or []:
        print(row)


if __name__ == "__main__":
    print("### tight scaffold, exhaustive enumeration\n")
    enum_table()
    print("\n### tight scaffold, gradient\n")
    gradient_table("tight")
    print("\n### wide scaffold, gradient\n")
    gradient_table("wide")
    print("\n### economy\n")
    economy_table()
