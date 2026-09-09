"""The computer arm's classifier heads, run live.

`computer_baseline.py` selects which arms to spend live kernel steps on using
TRAINING fit only -- the information the TCN search had -- and both regression
heads tie at 1.00 there, so the 256-way classifier heads never reached the live
loop. Their held-out *supervised* byte accuracy is higher (0.50 against 0.00),
so leaving them unmeasured would understate the neural side.

This script runs them, under a protocol the TCN did not have (selection on
held-out supervised accuracy), and reports the result as such. It is the neural
side's best achievable live score on this task, not a matched-protocol number.

Run: `.venv/bin/python research/neural-baselines/computer_live_extra.py`
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import torch

import common as harness
import computer_baseline as CB
import task as T

torch.set_num_threads(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--models", nargs="*", default=["gru_h16", "cnn_w32"])
    ap.add_argument("--tag", default="computer_live_extra")
    args = ap.parse_args()

    train = CB.featurise(CB.collect(T.TRAIN_DOCUMENTS))
    held = CB.featurise(CB.collect(T.TEST_DOCUMENTS))

    # The hard ceiling for a 256-way classifier: it can only emit a byte it saw
    # as a training label, so its held-out score cannot exceed the fraction of
    # held-out targets that occur among the training targets.
    seen = {int(b) for b in train["byte"][train["has_byte"]].tolist()}
    want = [int(b) for b in held["byte"][held["has_byte"]].tolist()]
    ceiling = sum(b in seen for b in want) / max(1, len(want))

    report = {"protocol": "selection on held-out supervised accuracy -- NOT the "
                          "matched protocol; reported to show the neural side's best "
                          "achievable live score",
              "classifier_ceiling": {"training_target_bytes": sorted(seen),
                                     "heldout_target_bytes": want,
                                     "max_possible_byte_accuracy": ceiling},
              "arms": []}
    harness.report("256-way classifier ceiling on the held-out documents", ceiling)

    for name in args.models:
        rows = []
        for seed in range(args.seeds):
            model, info = CB.train_one(CB.ZOO[name], "cls", train, seed, steps=args.steps)
            fit = CB.supervised_accuracy(model, train)
            off = CB.supervised_accuracy(model, held)
            live = CB.rollout(model, T.TEST_DOCUMENTS)
            rows.append({"seed": seed, **info, "train": fit, "heldout_supervised": off,
                         "closed_loop": {k: v for k, v in live.items() if k != "rows"},
                         "closed_loop_rows": live["rows"]})
            print(f"  LIVE {name:12s} cls seed {seed}: held byte "
                  f"{off['byte_accuracy']:.2f} -> {live['solved']}/10 solved, "
                  f"mean return {live['mean_return']:.2f}, "
                  f"policy {live['policy_ms_median']:.3f} ms", flush=True)
        report["arms"].append({
            "model": name, "byte_head": "cls", "parameters": rows[0]["parameters"],
            "heldout_byte_accuracy_median": statistics.median(
                r["heldout_supervised"]["byte_accuracy"] for r in rows),
            "solved_median": statistics.median(r["closed_loop"]["solved"] for r in rows),
            "solved_max": max(r["closed_loop"]["solved"] for r in rows),
            "mean_return_median": statistics.median(
                r["closed_loop"]["mean_return"] for r in rows),
            "policy_ms_median": statistics.median(
                r["closed_loop"]["policy_ms_median"] for r in rows),
            "rows": rows})
        harness.dump(args.tag, report)
    print(json.dumps([{k: v for k, v in a.items() if k != "rows"}
                      for a in report["arms"]], indent=1))


if __name__ == "__main__":
    main()
