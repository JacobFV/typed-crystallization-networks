"""Regenerate every table in RESULTS.md directly from the files in `out/`.

Nothing in the results document is typed by hand from a log; `python
research/semantic-library/tables.py` prints the tables and the raw numbers they
come from.
"""
from __future__ import annotations

import json
from pathlib import Path

import _paths  # noqa: F401

import armlib
import evaltasks

OUT = Path(__file__).resolve().parent / "out"


def load(name, default=None):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else default


def l1_table():
    grad_t = load("gradient_tight.json", {}).get("summary", [])
    grad_w = load("gradient_wide.json", {}).get("summary", [])
    gt = {r["arm"]: r for r in grad_t}
    gw = {r["arm"]: r for r in grad_w}
    print("\n## L1 (maj(a,b,c) xor maj(d,e,f)) -- tight scaffold enumeration + gradient\n")
    print("| arm | module | space | evaluated | exhausted | certificate | conforming |"
          " tight | median steps | median acc | wide |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in armlib.L1_ARMS:
        e = load(f"enum_tight_{arm}.json")
        if e is None:
            continue
        g, w = gt.get(arm, {}), gw.get(arm, {})
        print(f"| `{arm}` | {e.get('module') or '--'} | {e['space_size']:,} |"
              f" {e['evaluated']:,} | {e['exhausted']} | `{e['certificate']}` |"
              f" **{e['conforming']}** |"
              f" {g.get('solved', '')}/{g.get('seeds', '')} |"
              f" {g.get('median_steps_to_conformant', '--')} |"
              f" {g.get('median_accuracy', float('nan')):.4f} |"
              f" {w.get('solved', '')}/{w.get('seeds', '')} |")
    print("\nconstant baseline 0.5000, uniform-random baseline 0.5000 (both exact).")


def heldout_table():
    enum = load("heldout_enum.json", [])
    grad = load("heldout_gradient.json", {}).get("summary", [])
    ge = {(r["task"], r["arm"]): r for r in grad}
    if not enum:
        return
    print("\n## Held-out: mined from the remaining five tasks, evaluated on the sixth\n")
    print("| task | arm | space | evaluated | exhausted | certificate | conforming |"
          " seeds | median acc | constant |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for t in evaltasks.LOO_TASKS + tuple(evaltasks.CONTROL_TASKS):
        for a in armlib.HELDOUT_ARMS:
            r = next((x for x in enum if x["task"] == t and x["arm"] == a), None)
            if r is None:
                continue
            g = ge.get((t, a), {})
            print(f"| `{t}` | `{a}` | {r['space_size']:,} | {r['evaluated']:,} |"
                  f" {r['exhausted']} | `{r['certificate']}` | **{r['conforming']}** |"
                  f" {g.get('solved', '')}/{g.get('seeds', '')} |"
                  f" {g.get('median_accuracy', float('nan')):.4f} |"
                  f" {r['constant_baseline']:.4f} |")


def mined_table():
    s = load("mine_summary.json") or {}
    checks = load("checks.json", {})
    per = checks.get("per_corpus", {})
    print("\n## What each corpus proposed\n")
    print("| corpus | identity | entries | eligible | rank-1 is the family window? |"
          " best window rank | rank-1 arity/nodes | rank-1 digest |")
    print("|---|---|---|---|---|---|---|---|")
    for name, v in sorted(per.items()):
        r1 = v.get("rank1") or {}
        print(f"| `{name}` | {v['identity']} | {v['entries']} | {v['eligible']} |"
              f" **{v['rank1_is_window']}** | {v['best_window_rank']} |"
              f" {r1.get('arity')}/{r1.get('nodes')} | `{r1.get('digest', '')[:12]}` |")


def sweep_table():
    for name in ("maj_minall_sem", "maj_trace_sem"):
        d = load(f"sweep_{name}.json")
        if d is None:
            continue
        print(f"\n## Sweep: every eligible arity-3 pooled class of `{name}` on L1's tight scaffold\n")
        print(f"{d['classes_swept']} classes swept of {d['eligible_total']} eligible; "
              f"{d['reached_ceiling']} reach 144 conforming; "
              f"{d['any_conforming']} reach any conforming program.")
        print("\n| rank | window? | nodes | conforming | exhausted | certificate |")
        print("|---|---|---|---|---|---|")
        for r in d["rows"]:
            print(f"| {r['rank']} | {'**yes**' if r['is_window'] else 'no'} | {r['nodes']} |"
                  f" **{r['conforming']}** | {r['exhausted']} | `{r['certificate']}` |")


def sensitivity_table():
    d = load("sensitivity.json")
    if d is None:
        return
    print(f"\n## Rule sensitivity (semantic identity, C-minall): rank-1 is MAJ3 in "
          f"{d['rank1_is_maj3']} of {d['grid']} settings\n")
    print("| max_nodes | max_holes | min_tasks | eligible | rank-1 is MAJ3 | best MAJ3 rank |")
    print("|---|---|---|---|---|---|")
    for r in d["rows"]:
        print(f"| {r['max_nodes']} | {r['max_holes']} | {r['min_tasks']} | {r['eligible']} |"
              f" {r['rank1_is_maj3']} | {r['best_maj3_rank']} |")


def cost_table():
    c = load("checks.json", {}).get("mining_cost", {})
    off = load("off_corpora.json", {})
    maj_bands = load("../premin-abstraction/out/corpora.json")
    print("\n## Cost\n")
    print(json.dumps(c, indent=2, sort_keys=True))
    if off:
        dfs = sum(b["dfs_expanded"] for t in off["tasks"] for b in t["bands"].values())
        print(f"off-family corpus build: {dfs:,} DFS nodes, "
              f"{off['wall_seconds']:.0f} wall seconds")
    if maj_bands:
        dfs = sum(b["dfs_expanded"] for t in maj_bands["tasks"] for b in t["bands"].values())
        print(f"majority-family corpus build (§46's, reused): {dfs:,} DFS nodes")


if __name__ == "__main__":
    mined_table()
    l1_table()
    heldout_table()
    sweep_table()
    sensitivity_table()
    cost_table()
