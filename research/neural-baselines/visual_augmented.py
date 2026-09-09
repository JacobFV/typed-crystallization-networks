"""The visual arm's CNN with the one fix its failure mode calls for.

`visual_bounds.py` measures why the matched CNN generalizes badly: **81.2% of the
raw 3x3 pixel contexts on the 12 held-out screens never occur on the 6 training
screens**, and a lookup table over raw values transfers at 0.9824 against a
0.9815 majority -- an advantage of 0.0009. Over the *equality* signature
(same-as-left / same-as-up / same-as-up-left) the same table transfers at 1.000.
The information is in the equality relation, which the TCN scaffold has as a
primitive (S0's `same`) and a CNN over raw values has to learn.

So the practitioner's move is a label-preserving colour augmentation: recolour
each training screen through a random bijection of its palette. Rectangles,
corners and the hierarchy are all invariant under it, so no label changes.

**This arm is a DECLARED STRUCTURAL HINT and is reported apart from the matched
tables.** Telling the model that the answer does not depend on colour identity is
telling it something the matched arm had to discover -- the same category of help
`research/baselines/RESULTS.md` §6 refused when it declined Fourier features on
the mixed fixture. It is run and reported because the size of the gap it closes
is the informative number, not because it belongs in the matched comparison.

Run: `.venv/bin/python research/neural-baselines/visual_augmented.py`
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import torch
import torch.nn as nn

import common as harness
import visual_baseline as VB

torch.set_num_threads(1)


def recolour(pixels, generator):
    """A random bijection of each channel's byte values, applied per image.

    Per-channel bijections preserve equality of RGB triples exactly, so every
    corner, extent and parent link in the screen is unchanged.
    """
    out = pixels.clone()
    for i in range(out.shape[0]):
        for c in range(3):
            perm = torch.randperm(256, generator=generator).float() / 255.
            index = (out[i, c] * 255).round().long().clamp(0, 255)
            out[i, c] = perm[index]
    return out


def train_augmented(name, data, steps, seed, lr=3e-3, pos_weight=40.0, batch=6):
    torch.manual_seed(seed)
    model = VB.ZOO[name]()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight), reduction="none")
    ce = nn.CrossEntropyLoss(reduction="none")
    n = data["pixels"].shape[0]
    generator = torch.Generator().manual_seed(seed)
    started = time.perf_counter()
    for _ in range(steps):
        idx = (torch.arange(n) if n <= batch
               else torch.randint(0, n, (batch,), generator=generator))
        pixels = recolour(data["pixels"][idx], generator)
        opt.zero_grad()
        c, wl, hl = model(pixels)
        loss = bce(c, data["corner"][idx]).mean()
        at = data["corner"][idx] > 0
        if at.any():
            loss = loss + (ce(wl, data["w"][idx])[at].mean() + ce(hl, data["h"][idx])[at].mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    return model, {"train_seconds": time.perf_counter() - started,
                   "final_loss": float(loss.detach()), "steps": steps, "lr": lr,
                   "train_screens": n, "batch": min(batch, n),
                   "parameters": sum(p.numel() for p in model.parameters())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--models", nargs="*", default=["cnn_w32_d5"])
    ap.add_argument("--screens", type=int, nargs="*", default=[6, 48])
    ap.add_argument("--tag", default="visual_augmented")
    args = ap.parse_args()

    train = [VB.screen(s, "train") for s in VB.TRAIN_SEEDS]
    val = [VB.screen(s, "validation") for s in VB.VAL_SEEDS]
    test = [VB.screen(VB.TEST_FIRST + i, "test") for i in range(VB.TEST_SCREENS)]
    pool = [VB.screen(s, "train") for s in range(max(args.screens))]

    report = {"declared": "colour-bijection augmentation is a structural hint; "
                          "reported apart from the matched tables",
              "trivial_reference_test": VB.trivial_references(test), "arms": []}
    for n in args.screens:
        data = VB.stack(train if n == 6 else pool[:n])
        for name in args.models:
            rows = []
            for seed in range(args.seeds):
                model, info = train_augmented(name, data, args.steps, seed)
                threshold, _ = VB.choose_threshold(model, val)
                te = VB.evaluate(model, test, threshold)["totals"]
                rows.append({"seed": seed, **info, "threshold": threshold, "test": te})
                print(f"  augmented {n:3d} screens {name:14s} seed {seed}: "
                      f"thr {threshold:+.1f} rects {te['rects_exactly_right']}/"
                      f"{te['widgets_in_probe']} links {te['parent_links_correct']}/"
                      f"{te['widgets_in_probe']} trees {te['trees_exact']}/12 "
                      f"corner P{te['corner_precision']:.2f}/R{te['corner_recall']:.2f} "
                      f"extent {te['extent_accuracy']:.2f} "
                      f"({info['train_seconds']:.0f}s)", flush=True)
            report["arms"].append({
                "train_screens": n, "model": name, "parameters": rows[0]["parameters"],
                "test_link_accuracy_median": statistics.median(
                    r["test"]["link_accuracy"] for r in rows),
                "test_rect_accuracy_median": statistics.median(
                    r["test"]["rect_accuracy"] for r in rows),
                "test_trees_exact_median": statistics.median(
                    r["test"]["trees_exact"] for r in rows),
                "corner_recall_median": statistics.median(
                    r["test"]["corner_recall"] for r in rows),
                "corner_precision_median": statistics.median(
                    r["test"]["corner_precision"] for r in rows),
                "extent_accuracy_median": statistics.median(
                    r["test"]["extent_accuracy"] for r in rows),
                "train_seconds_median": statistics.median(r["train_seconds"] for r in rows),
                "rows": rows})
            harness.dump(args.tag, report)
    print(json.dumps([{k: v for k, v in a.items() if k != "rows"}
                      for a in report["arms"]], indent=1))


if __name__ == "__main__":
    main()
