"""Run every arm at a budget, over seeds, with the compute equalized.

Two phases per seed, because the padding target is not known in advance:

1. run the crystallizing arms and record each one's realised `optimizer.step()`
   total (retraining inside rolled-back trials included);
2. run plain argmax twice, padded with extra optimizer steps on the same
   objective the crystallizer retrains on, once to each scheduler arm's total.
   `argmax0` is the unpadded control and always spends strictly less.

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
PYTHON = sys.executable


def invoke(script, args):
    out = subprocess.run([PYTHON, str(HERE / script), *args], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(f"{script} {args} failed:\n{out.stderr[-4000:]}")
    return json.loads(out.stdout)


def mixed_call(arm, seed, steps, noise, extra=0):
    return invoke("run_mixed.py", ["--arm", arm, "--seed", str(seed), "--steps", str(steps),
                                   "--noise", str(noise), "--extra", str(extra)])


def joint_call(arm, seed, episodes, extra=0):
    return invoke("run_joint.py", ["--arm", arm, "--seed", str(seed),
                                   "--episodes", str(episodes), "--extra", str(extra)])


def seed_block(task, seed, budget, noise):
    call = (lambda arm, extra=0: mixed_call(arm, seed, budget, noise, extra)) if task == "mixed" \
        else (lambda arm, extra=0: joint_call(arm, seed, budget, extra))
    base = budget
    rows = []
    totals = {}
    schedulers = ["entropy", "perturbation"] + (["perturbation-total-guard"] if task == "joint" else [])
    for arm in schedulers:
        r = call(arm)
        totals[arm] = r["total_steps"]
        rows.append(r)
    for arm in ("entropy", "perturbation"):
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
    ap.add_argument("--noise", type=float, default=0.5, help="mixed only; 0 = shipped fixture")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out")
    args = ap.parse_args()

    out = Path(args.out) if args.out else HERE / "results" / f"{args.task}-{args.budget}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(seed_block, args.task, seed, args.budget, args.noise)
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
