"""Regenerate every table in RESULTS.md from `out/*.json`.

Nothing in RESULTS.md is transcribed by hand; run this and diff.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
DEPTHS = ["1", "2", "3", "4", "6", "8"]


def load(name):
    return json.loads((OUT / f"{name}.json").read_text())


def audit_table():
    a = load("audit")
    print("### width-entry sites\n")
    print("| # | site | class | what it compares |")
    print("|---:|---|---|---|")
    for i, s in enumerate(a["sites"], 1):
        print(f"| {i} | `{s['cite']}` | {s['klass']} | {s['what']} |")
    print("\ncounts:", json.dumps(a["site_counts"]))
    print("\n### executable demonstrations\n")
    print("| probe | outcome |")
    print("|---|---|")
    for k, v in a["demonstrations"].items():
        if isinstance(v, dict) and "raised" in v:
            got = (f"`{v['type']}: {v['message']}` at `{v['at']}`" if v["raised"]
                   else "no error")
        else:
            got = "`" + json.dumps(v) + "`"
        print(f"| `{k}` | {got} |")


def transfer_table():
    b = load("baselines")["held_out"]
    t = load("transfer")
    e = load("exhaust")
    r = load("record")
    w = load("wrong_schema")
    print("\n### held-out return, 64 episodes, 0-4 scale\n")
    head = "| arm | " + " | ".join("d" + d for d in DEPTHS) + " |"
    print(head)
    print("|---|" + "---|" * len(DEPTHS))
    for fit_depth, rows in sorted(t.items()):
        for row in rows:
            cells = " | ".join(f"**{row['per_depth'][d]['mean']:.4f}**" for d in DEPTHS)
            print(f"| A: interpreter, selections fitted at d={fit_depth} | {cells} |")
    for row in w["transfer"]:
        cells = " | ".join(f"{row['per_depth'][d]['mean']:.4f}" for d in DEPTHS)
        print(f"| F: wrong schema, same vector, fitted at d=1 | {cells} |")
    best = max(r["transfer"], key=lambda x: sum(x["per_depth"][d]["mean"] for d in DEPTHS))
    cells = " | ".join(f"{best['per_depth'][d]['mean']:.4f}" for d in DEPTHS)
    print(f"| B: record scaffold (best of {len(r['transfer'])} carried) | {cells} |")
    for label, key in (("best constant", "best_constant"),):
        cells = " | ".join(f"{b[d][key]:.4f}" for d in DEPTHS)
        print(f"| {label} | {cells} |")
    cells = " | ".join(f"{b[d]['uniform_random']['mean']:.4f}" for d in DEPTHS)
    print(f"| uniform random | {cells} |")
    cells = " | ".join(f"{e[d]['distribution']['mean']:.4f}" for d in DEPTHS)
    print(f"| mean over the whole 272-program space (arm C/E) | {cells} |")
    cells = " | ".join(f"{sorted(e[d]['all_returns'])[-2]:.4f}" for d in DEPTHS)
    print(f"| second-best of the 272 (arm E) | {cells} |")

    print("\n### certificates\n")
    print("| sweep | space | evaluated | exhausted | conforming @4.0 | certificate |")
    print("|---|---:|---:|---|---:|---|")
    f = load("fit")
    for d, v in sorted(f.items()):
        print(f"| A fit, interpreter, depth {d}, 16 train episodes | {v['space_size']} | "
              f"{v['evaluated']} | {v['exhausted']} | {v['conforming']} | `{v['certificate']}` |")
    for d in DEPTHS:
        v = e[d]
        print(f"| E held-out exhaustion, interpreter, depth {d} | {v['space_size']} | "
              f"{v['evaluated']} | {v['exhausted']} | {v['conforming']} | `{v['certificate']}` |")
    v = w["fit"]
    print(f"| F fit, wrong schema, depth 1 | {v['space_size']} | {v['evaluated']} | "
          f"{v['exhausted']} | {v['conforming']} | `{v['certificate']}` |")
    for d in DEPTHS:
        v = w["exhaust"][d]
        print(f"| F held-out exhaustion, wrong schema, depth {d} | {v['space_size']} | "
              f"{v['evaluated']} | {v['exhausted']} | {v['conforming']} | `{v['certificate']}` |")
    v = r["fit"]
    print(f"| B fit, record scaffold, depth 1 | {v['space_size']} | {v['evaluated']} | "
          f"{v['exhausted']} | {v['conforming']} | `{v['certificate']}` |")


def size_table():
    s = load("size")
    d = load("difficulty")
    print("\n### one schema, six artifacts\n")
    print("| depth | `program` fields | scaffold nodes | frozen digest | description bits | execution cost |")
    print("|---:|---:|---:|---|---:|---:|")
    for k in DEPTHS:
        v = s["per_depth"][k]
        print(f"| {k} | {v['program_fields']} | {v['scaffold_nodes']} | `{v['digest']}` | "
              f"{v['description_bits']} | {v['execution_cost']:.0f} |")
    print(f"\nselection vector: {s['selection_vector_entries']} integers, "
          f"{s['selection_vector_bits']:.2f} bits, identical at all six depths; "
          f"{s['distinct_digests']} distinct frozen artifacts.")
    print("\n### achieved difficulty, held-out episodes\n")
    print("| depth | distinct circuits | distinct final tables | relevant-input histogram | majority fraction |")
    print("|---:|---:|---:|---|---:|")
    for k in DEPTHS:
        v = d[k]
        print(f"| {k} | {v['distinct_circuits']}/64 | {len(v['distinct_final_tables'])} | "
              f"`{json.dumps(v['relevant_inputs_histogram'])}` | {v['majority_fraction']:.4f} |")


def hardened_table():
    h = load("hardened")
    print("\n### arm D — the hardened artifact itself\n")
    print(f"depth-1 frozen artifact `{h['frozen_digest']}` offered each depth's observation:\n")
    print("| depth | outcome |")
    print("|---:|---|")
    for k in DEPTHS:
        v = h["offered"][k]
        print(f"| {k} | " + (f"`{v['type']}: {v['message']}` at `{v['at']}`" if v["raised"]
                             else "executes") + " |")
    c = load("crosscheck")
    print(f"\ncross-check against `tcn.search.enumerate_environment`: "
          f"all_agree = {c['all_agree']}, {json.dumps(c['agrees'])}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["audit", "transfer", "size", "hardened"]
    for name in which:
        {"audit": audit_table, "transfer": transfer_table,
         "size": size_table, "hardened": hardened_table}[name]()
