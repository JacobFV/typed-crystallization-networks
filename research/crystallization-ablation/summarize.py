"""Aggregate ablation JSONL records into the markdown tables used by RESULTS.md."""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

LABEL = {
    "A": "**A** full crystallizer",
    "B": "**B** naive argmax (budget-matched)",
    "B0": "B0 naive argmax (unpadded control)",
    "C": "**C** shuffled freeze order",
    "D": "**D** retrain_steps=0",
    "E": "**E** tolerance=inf",
    "F": "**F** perturbation selection (decisions first)",
    "FR": "**FR** perturbation-by-removal (DARTS-PT)",
    "F2": "F2 perturbation selection (plumbing first)",
    "G": "G accept-all (no transaction)",
    "H": "**H** A with conformance check disabled",
}
ORDER = ["A", "H", "B", "B0", "C", "D", "E", "F", "FR", "F2", "G"]
COLS = ["degradation", "disconnected remaining region", "runtime conformance",
        "nonfinite loss", "invalid trial"]


def ms(xs, digits=3):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    if len(set(xs)) == 1:
        return f"{xs[0]:.{digits}g}"
    return f"{st.mean(xs):.{digits}g} +/- {st.stdev(xs):.{digits}g}"


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def evals(r):
    return r.get("loss_evaluations", 0) + r.get("selection_sweep_evaluations", 0)


def mixed_table(rows, steps):
    g_all = defaultdict(list)
    for r in rows:
        g_all[r["arm"]].append(r)
    out = [f"\n**mixed synthesis, base training steps = {steps}, {len(g_all['A'])} seeds**\n",
           "| arm | exact conformance | final task loss | frozen nodes | optimizer steps | objective evals | wall (s) |",
           "|---|---|---|---|---|---|---|"]
    for arm in ORDER:
        g = g_all.get(arm)
        if not g:
            continue
        out.append("| {} | {}/{} | {} | {} | {} | {} | {} |".format(
            LABEL[arm], sum(1 for r in g if r["exact_conformance"]), len(g),
            ms([r["loss"] for r in g], 3),
            ms([r["frozen_fraction"] for r in g], 3),
            ms([r["total_steps"] for r in g], 4),
            ms([evals(r) for r in g], 4),
            ms([r["wall"] for r in g], 3)))
    return "\n".join(out)


def joint_table(rows, episodes):
    g_all = defaultdict(list)
    for r in rows:
        g_all[r["arm"]].append(r)
    out = [f"\n**joint logic task, {episodes} training episodes, {len(g_all['A'])} seeds, "
           "16 held-out test episodes (max return 4)**\n",
           "| arm | fully frozen + exportable | det. mean return | frozen-program return | frozen nodes | optimizer steps | objective evals | wall (s) |",
           "|---|---|---|---|---|---|---|---|"]
    for arm in ORDER:
        g = g_all.get(arm)
        if not g:
            continue
        out.append("| {} | {}/{} | {} | {} | {} | {} | {} | {} |".format(
            LABEL[arm], sum(1 for r in g if r["exact_conformance"]), len(g),
            ms([r["post_crystallization"]["mean_return"] if r["post_crystallization"] else None for r in g], 4),
            ms([r["frozen_agent_mean_return"] for r in g], 4),
            ms([r["frozen_fraction"] for r in g], 3),
            ms([r["total_steps"] for r in g], 4),
            ms([evals(r) for r in g], 4),
            ms([r["wall"] for r in g], 3)))
    return "\n".join(out)


def reason_table(rows, label):
    by_arm = defaultdict(Counter)
    trials = Counter()
    runs = Counter()
    for r in rows:
        if r["arm"] in ("B", "B0"):
            continue
        runs[r["arm"]] += 1
        for e in r.get("freeze_events", []):
            trials[r["arm"]] += 1
            by_arm[r["arm"]][("accept" if e["accepted"] else "rollback",
                              e["reason"].split(":")[0])] += 1
    out = [f"\n**{label}: freeze-trial outcomes**\n",
           "| arm | runs | freeze trials | accepted | degradation | disconnected remaining region | runtime conformance | nonfinite loss | invalid trial |",
           "|---|---|---|---|---|---|---|---|---|"]
    for arm in ORDER:
        if arm not in runs:
            continue
        c = by_arm[arm]
        acc = sum(v for k, v in c.items() if k[0] == "accept")
        cells = [str(c.get(("rollback", col), 0)) for col in COLS]
        out.append("| {} | {} | {} | {} | {} |".format(
            LABEL[arm], runs[arm], trials[arm], acc, " | ".join(cells)))
    return "\n".join(out)


def deferral_analysis(rows, arm="A"):
    deferred_then_frozen = never_frozen = runs_with_rollback = total_runs = 0
    for r in rows:
        if r["arm"] != arm:
            continue
        total_runs += 1
        rejected = {e["node"] for e in r.get("freeze_events", []) if not e["accepted"]}
        accepted = {e["node"] for e in r.get("freeze_events", []) if e["accepted"]}
        if rejected:
            runs_with_rollback += 1
        deferred_then_frozen += len(rejected & accepted)
        never_frozen += len(rejected - accepted)
    return {"runs": total_runs, "runs_with_at_least_one_rollback": runs_with_rollback,
            "nodes_rolled_back_then_frozen_later": deferred_then_frozen,
            "nodes_rolled_back_and_never_frozen": never_frozen}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    for f in args.files:
        rows = load(f)
        task = rows[0]["task"]
        tag = rows[0].get("steps") if task == "mixed" else rows[0].get("episodes")
        print(f"\n<!-- source: {f} -->")
        print(mixed_table(rows, tag) if task == "mixed" else joint_table(rows, tag))
        print(reason_table(rows, f"{task}, {tag}"))
        print("\nDeferral analysis, arm A: " + json.dumps(deferral_analysis(rows)))


if __name__ == "__main__":
    main()
