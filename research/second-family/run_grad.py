"""The gradient panel, at the configuration this family's arity actually allows.

§52 ran 24 seeds x 400 steps.  It could: its module was 3-ary, so the soft
program was 336 candidates wide at its widest node.  This family's window is
**4-ary**, which makes `compact_scaffold`'s output node 1,416 candidates wide
over 32 rows, and one 400-step seed costs **~240 wall-seconds** on this
machine.  The panel is therefore run at **8 seeds x 200 steps** on the
`C-trace` band and on the arm table's own task, and the *achieved*
configuration is what is reported -- never the one §52 used.

Enumeration, not gradient, carries this track's claims: every one of them has
`evaluated == space_size` and a certificate beside it.  This panel exists so
that an accuracy figure with its constant and random baselines sits beside the
conforming counts, as §52's table did.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _paths  # noqa: F401

import evaltasks
import run_arms

OUT = Path(__file__).resolve().parent / "out"

ARMS = ("arm1_none", "arm2_syntactic", "arm2s_semantic", "arm3_authored",
        "arm4_wrong_authored", "arm4s_runnerup")
BAND = "C-trace"
SEEDS = 8
STEPS = 200


def main(task=None):
    task = task or evaltasks.LATER_TASK
    OUT.mkdir(exist_ok=True)
    jobs = [(BAND, a, task, s, STEPS) for a in ARMS for s in range(SEEDS)]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("TCN_WORKERS", "8"))) as ex:
        recs = list(ex.map(run_arms._grad, jobs))
    summary = []
    for a in ARMS:
        rows = [x for x in recs if x["arm"] == a]
        accs = sorted(x["accuracy"] for x in rows)
        summary.append({
            "band": BAND, "arm": a, "task": task, "seeds": len(rows),
            "steps": STEPS, "solved": sum(1 for x in rows if x["conformant"]),
            "median_accuracy": accs[len(accs) // 2],
            "mean_accuracy": sum(accs) / len(accs), "best_accuracy": accs[-1],
            "worst_accuracy": accs[0],
            "constant_baseline": evaltasks.constant_baseline(
                evaltasks.COMPACT_TASKS[task]),
            "random_baseline": 0.5})
        print(json.dumps(summary[-1], sort_keys=True), flush=True)
    (OUT / "arms_gradient.json").write_text(json.dumps(
        {"band": BAND, "seeds": SEEDS, "steps": STEPS, "task": task,
         "summary": summary, "records": recs}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
