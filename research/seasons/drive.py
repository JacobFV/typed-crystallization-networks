"""Run every arm at a budget, over seeds, with the compute equalized.

Two phases per seed, because the padding target is not known in advance:

1. run the crystallizing arms and the populations, recording each one's realised
   `optimizer.step()` total -- retraining inside rolled-back trials included, and
   every summer's retraining included, and for a population summed over members;
2. run plain argmax, padded with extra optimizer steps on the same objective the
   crystallizer retrains on, up to each named arm's total.  `argmax0` is the
   unpadded control and always spends strictly less than any arm.

`argmax-restarts` is the population's own control: P independent argmax members
with no selection, scored best-of-P at the same per-member budget.

Each run is a separate subprocess so that torch state, RNG and the counting
wrappers cannot leak between arms.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
PYTHON = sys.executable

MIXED_SCHEDULERS = ("shipped", "shipped-cap", "prune", "seasons", "seasons-noblock")
JOINT_SCHEDULERS = ("shipped", "seasons")
MIXED_PADDED = ("shipped", "seasons")
JOINT_PADDED = ("shipped", "seasons")
# (member arm, selection rule) pairs carried as populations.
MIXED_POPULATIONS = (("seasons", "pareto"), ("seasons", "none"),
                     ("argmax", "pareto"), ("argmax", "none"))
JOINT_POPULATIONS = (("seasons", "pareto"), ("argmax", "none"))


def invoke(script, args):
    out = subprocess.run([PYTHON, str(HERE / script), *args], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(f"{script} {args} failed:\n{out.stderr[-4000:]}")
    return json.loads(out.stdout)


def seed_block(task, seed, budget, noise, members, generations):
    rows = []
    totals = {}
    if task == "mixed":
        def call(arm, extra=0):
            return invoke("run_mixed.py", ["--arm", arm, "--seed", str(seed), "--steps", str(budget),
                                           "--noise", str(noise), "--extra", str(extra)])

        def pop_call(arm, select):
            return invoke("population.py", ["--arm", arm, "--seed", str(seed), "--steps", str(budget),
                                            "--members", str(members), "--generations", str(generations),
                                            "--select", select, "--noise", str(noise)])
        schedulers, padded, populations = MIXED_SCHEDULERS, MIXED_PADDED, MIXED_POPULATIONS
    else:
        def call(arm, extra=0):
            return invoke("run_joint.py", ["--arm", arm, "--seed", str(seed),
                                           "--episodes", str(budget), "--extra", str(extra)])

        def pop_call(arm, select):
            return invoke("population_joint.py", ["--arm", arm, "--seed", str(seed),
                                                  "--episodes", str(budget), "--members", str(members),
                                                  "--generations", str(generations),
                                                  "--select", select, "--noise", str(noise)])
        schedulers, padded, populations = JOINT_SCHEDULERS, JOINT_PADDED, JOINT_POPULATIONS

    for arm in schedulers:
        r = call(arm)
        totals[arm] = r["total_steps"]
        rows.append(r)
    for arm, select in populations:
        r = pop_call(arm, select)
        name = "argmax-restarts" if (arm, select) == ("argmax", "none") else f"population/{arm}/{select}"
        r["arm"] = name
        totals[name] = r["total_steps"]
        rows.append(r)
    base = rows[0]["base_steps"] if "base_steps" in rows[0] else budget
    for arm in list(padded) + [f"population/seasons/pareto"]:
        if arm not in totals:
            continue
        r = call("argmax", max(0, totals[arm] - base))
        r["arm"] = f"argmax@{arm}"
        r["budget_reference_total_steps"] = totals[arm]
        rows.append(r)
    r = call("argmax")
    r["arm"] = "argmax0"
    rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["mixed", "joint"])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--budget", type=int, required=True,
                    help="mixed: base training steps. joint: training episodes.")
    ap.add_argument("--noise", type=float, default=0.5, help="0 = the shipped deterministic fixture")
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out")
    args = ap.parse_args()

    out = Path(args.out) if args.out else HERE / "results" / f"{args.task}-{args.budget}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(seed_block, args.task, seed, args.budget, args.noise,
                               args.members, args.generations)
                   for seed in range(args.seeds)]
        with out.open("w") as fh:
            for i, future in enumerate(futures):
                for row in future.result():
                    fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(f"seed {i} done", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
