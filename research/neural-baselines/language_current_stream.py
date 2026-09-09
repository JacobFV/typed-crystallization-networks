"""The language task on the POST-AUDIT stream, where no TCN number exists yet.

FINDINGS §39: `context_free_language` was re-drawn by §24, string length moved
from 2-16 to 10-22, and because the language-capability track holds out *length*
its training split is empty on today's default stream. So **the language
capability has never been measured on the post-audit distribution**, and §39's
instruction is not to quote §19's 0.9986 beside anything trained there.

This script therefore does not compare. It measures one side: what a small
neural baseline reaches when it is *trained* on the post-audit stream, with the
track's own holdout discipline (hold out string length) transposed onto the new
length range, and with the trivial and oracle references beside it. It is
reported so that whoever measures the TCN side on this distribution has a
reference point to land against.

Split, mirroring `run_stage_b.py`'s shape on the new range:
  train  the first 24 episodes of `dataset(1200, seed0=0, split='train')`
         whose length is in {10, 12, 14}
  val    the next 120 of the same lengths
  test   every episode of `dataset(1200, seed0=100000, split='test')` whose
         length is in {16, 18, 20, 22} -- lengths never trained on

Run: `.venv/bin/python research/neural-baselines/language_current_stream.py`
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
import langdata
import language_baseline as LB

torch.set_num_threads(1)

TRAIN_LENGTHS = (10, 12, 14)
TEST_LENGTHS = (16, 18, 20, 22)


def counts_match(s):
    return s.count("(") == s.count(")")


def balanced(s):
    depth = 0
    for ch in s:
        depth += 1 if ch == "(" else -1
        if depth < 0:
            return False
    return depth == 0


def build():
    pool = langdata.dataset(1200, seed0=0, split="train", hardening=langdata.CURRENT)
    short = [e for e in pool if e["length"] in TRAIN_LENGTHS]
    test_pool = langdata.dataset(1200, seed0=100000, split="test",
                                 hardening=langdata.CURRENT)
    return {"train": short[:24], "val": short[24:144], "over_budget_train": short,
            "test": [e for e in test_pool if e["length"] in TEST_LENGTHS]}


def references(episodes):
    n = max(1, len(episodes))
    return {**harness.majority_reference([bool(e["label"]) for e in episodes]),
            "oracle_counting": sum(counts_match(e["string"]) == e["label"]
                                   for e in episodes) / n,
            "oracle_dyck": sum(balanced(e["string"]) == e["label"]
                               for e in episodes) / n,
            "lengths": sorted({e["length"] for e in episodes}),
            "distinct_strings": len({e["string"] for e in episodes})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--models", nargs="*",
                    default=["cnn_16x2", "cnn_32x3", "mlp_32", "transformer_d16"])
    ap.add_argument("--lrs", type=float, nargs="*", default=[0.01, 0.03])
    ap.add_argument("--tag", default="language_current_stream")
    args = ap.parse_args()

    sp = build()
    feats = {k: LB.featurise(v) for k, v in sp.items()}
    report = {"stream": "post-audit default (hardening not set)",
              "train_lengths": list(TRAIN_LENGTHS), "test_lengths": list(TEST_LENGTHS),
              "tcn_reference": "NONE -- the capability has not been measured on this "
                               "distribution (FINDINGS §39)",
              "splits": {k: {"n": len(v), **references(v)} for k, v in sp.items()},
              "arms": []}
    print(json.dumps(report["splits"]["test"], indent=1), flush=True)

    for budget, key, seeds in (("matched_24", "train", args.seeds),
                               ("over_budget", "over_budget_train", 2)):
        data = feats[key]
        for aux in (False, True):
            for name in args.models:
                rows = []
                for seed in range(seeds):
                    for lr in args.lrs:
                        model, info = LB.train_one(name, aux, data, seed,
                                                   steps=args.steps, lr=lr)
                        rows.append({"seed": seed, **info,
                                     "parameters": sum(p.numel() for p in model.parameters()),
                                     "train_acc": LB.accuracy(model, data),
                                     "val_acc": LB.accuracy(model, feats["val"]),
                                     "test_acc": LB.accuracy(model, feats["test"])})
                best_lr = max(args.lrs, key=lambda l: statistics.median(
                    r["train_acc"] for r in rows if r["lr"] == l))
                kept = [r for r in rows if r["lr"] == best_lr]
                arm = {"budget": budget, "aux": aux, "model": name, "lr": best_lr,
                       "lr_grid": list(args.lrs), "parameters": kept[0]["parameters"],
                       "train_acc_median": statistics.median(r["train_acc"] for r in kept),
                       "val_acc_median": statistics.median(r["val_acc"] for r in kept),
                       "test_acc_median": statistics.median(r["test_acc"] for r in kept),
                       "test_acc_max": max(r["test_acc"] for r in kept),
                       "rows": rows}
                report["arms"].append(arm)
                print(f"  {budget:12s} aux={int(aux)} {name:17s} lr {best_lr}: "
                      f"train {arm['train_acc_median']:.3f} val {arm['val_acc_median']:.3f} "
                      f"test {arm['test_acc_median']:.3f} "
                      f"(majority {report['splits']['test']['majority_constant']:.3f})",
                      flush=True)
                harness.dump(args.tag, report)

    best = max(report["arms"], key=lambda a: (a["val_acc_median"], -a["parameters"]))
    report["selected_on_validation"] = {k: best[k] for k in
                                        ("budget", "aux", "model", "lr", "parameters",
                                         "train_acc_median", "val_acc_median",
                                         "test_acc_median")}
    harness.dump(args.tag, report)
    print(json.dumps(report["selected_on_validation"], indent=1))


if __name__ == "__main__":
    main()
