"""Re-run the `H_par` rows after the chained-comparison defect was fixed.

Only the `H_par` task is recomputed; every other row of `heldout_enum.json`
and `heldout_gradient.json` is untouched and is the number from the original
run.  Which rows were replaced is recorded in `RESULTS.md`.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _paths  # noqa: F401

import armlib
import evaltasks
import run_heldout

OUT = Path(__file__).resolve().parent / "out"


def main():
    task = "H_par"
    workers = int(os.environ.get("TCN_WORKERS", "14"))
    enum = json.loads((OUT / "heldout_enum.json").read_text())
    enum = [r for r in enum if r["task"] != task]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(run_heldout._enum,
                             [(task, a) for a in armlib.HELDOUT_ARMS]))
    for r in rows:
        print(r["task"], r["arm"], "space", r["space_size"], "conforming",
              r["conforming"], r["exhausted"], r["certificate"],
              "constant", r["constant_baseline"], flush=True)
    (OUT / "heldout_enum.json").write_text(json.dumps(enum + rows, indent=2,
                                                      sort_keys=True))

    grad = json.loads((OUT / "heldout_gradient.json").read_text())
    grad["records"] = [x for x in grad["records"] if x["task"] != task]
    grad["summary"] = [x for x in grad["summary"] if x["task"] != task]
    jobs = [(task, a, s, 400) for a in armlib.HELDOUT_ARMS for s in range(24)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        recs = list(pool.map(run_heldout._grad, jobs))
    for a in armlib.HELDOUT_ARMS:
        rows_a = [x for x in recs if x["arm"] == a]
        hits = [x for x in rows_a if x["conformant"]]
        steps_to = sorted(x["first_conformant_step"] for x in hits
                          if x["first_conformant_step"] is not None)
        accs = sorted(x["accuracy"] for x in rows_a)
        grad["summary"].append({
            "task": task, "arm": a, "seeds": len(rows_a), "solved": len(hits),
            "median_steps_to_conformant": (steps_to[len(steps_to) // 2]
                                           if steps_to else None),
            "median_accuracy": accs[len(accs) // 2],
            "mean_accuracy": sum(accs) / len(accs), "best_accuracy": accs[-1],
            "constant_baseline": evaltasks.constant_baseline(
                evaltasks.COMPACT_TASKS[task]),
            "random_baseline": 0.5,
            "module_on_output_path": sum(1 for x in hits
                                         if x["module_on_output_path"]),
        })
        print(json.dumps(grad["summary"][-1], sort_keys=True), flush=True)
    grad["records"] += recs
    (OUT / "heldout_gradient.json").write_text(json.dumps(grad, indent=2,
                                                          sort_keys=True))


if __name__ == "__main__":
    main()
