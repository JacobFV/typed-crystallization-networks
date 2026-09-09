"""Run every (arm, seed) in a fresh subprocess, in two budget-equalization phases.

Phase 1 runs arm A (and every arm that needs no external budget). Phase 2 reads
arm A's realised optimizer-step total for each seed and gives arm B exactly that
many optimizer steps of plain soft training before the argmax rounding.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")


def call(script, out, **kwargs):
    cmd = [PY, str(HERE / script), "--single", "--out", str(out)]
    for k, v in kwargs.items():
        cmd += [f"--{k}", str(v)]
    subprocess.run(cmd, check=True, cwd=str(ROOT), capture_output=True)
    return json.loads(Path(out).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["mixed", "joint"])
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--arms", default="A,B,B0,C,D,E,F,FR,F2,G,H")
    ap.add_argument("--steps", type=int, default=300, help="mixed: base training steps")
    ap.add_argument("--episodes", type=int, default=160, help="joint: training episodes")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    script = "run_mixed.py" if args.task == "mixed" else "run_joint.py"
    base = {"steps": args.steps} if args.task == "mixed" else {"episodes": args.episodes}
    base_steps = args.steps if args.task == "mixed" else args.episodes
    tag = args.tag or (str(args.steps) if args.task == "mixed" else str(args.episodes))
    scratch = HERE / "results" / f"{args.task}-{tag}-raw"
    scratch.mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",")

    phase1 = [a for a in arms if a != "B"]
    jobs = [(a, s) for s in range(args.seeds) for a in phase1]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda j: call(script, scratch / f"{j[0]}_{j[1]}.json",
                                            arm=j[0], seed=j[1], **base), jobs))
    budgets = {r["seed"]: r["total_steps"] for r in rows if r["arm"] == "A"}
    print("arm A optimizer-step budgets:", budgets, flush=True)

    if "B" in arms:
        jobs = [(s, budgets[s] - base_steps) for s in range(args.seeds)]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows += list(pool.map(lambda j: call(script, scratch / f"B_{j[0]}.json",
                                                 arm="B", seed=j[0], extra=j[1], **base), jobs))
        for r in rows:
            r["budget_reference_total_steps"] = budgets.get(r["seed"])

    out = HERE / "results" / f"{args.task}-{tag}.jsonl"
    with out.open("w") as fh:
        for r in sorted(rows, key=lambda r: (r["seed"], r["arm"])):
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    sys.exit(main())
