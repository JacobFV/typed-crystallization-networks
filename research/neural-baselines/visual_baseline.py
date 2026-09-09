"""Matched neural baseline for FINDINGS §32/§33 -- a screenshot parsed to a hierarchy.

WHAT THE TCN ARTIFACT DOES, so the baseline can be held to the same contract.
`research/visual-ladder/rung3_root.py` runs three frozen modules at every one of
the 1,024 pixel addresses of a 32x32 RGB screen:

  S0   `same(a, b)`   -- two pixels have equal colour              (frozen, reused)
  S1'  `corner(pos)`  -- this pixel starts a widget                (searched: 400)
  S2'  `rect(pos)`    -- (x, y, w, h, own_key, parent_key)         (searched: 25)

and the scorer `score_with_root` then resolves the tree from those rows alone:
a row is the root when `parent_key == own_key`, otherwise its parent is the
smallest parsed rectangle containing the pixel at `(x - 1, y)`. That resolution
step is bookkeeping over the parse's own output, identical for both methods here.

MATCHING CONDITIONS.

* **Inputs.** The 3,072 raw RGB bytes of `observations['pixels']`, and nothing
  else. No baseline sees `probes['hierarchy']`, `probes['owner']`, the seed, the
  split, or the widget count as an input.
* **Outputs and scoring.** The CNN predicts, per pixel position, `corner`, `w`
  and `h`. `own_key` / `parent_key` are then packed from the raw image by the
  same declared, unsearched rule S2' uses (`pack` of the RGB triple at `pos`, and
  at `max(pos, 3) - 3` -- so the origin's parent pixel is itself and marks the
  root). Rows go through `rung3_root.score_with_root` unmodified, so the reported
  rectangles / links / trees are computed by the artifact's own scorer.
* **Supervision.** The `hierarchy` probe, which is exactly what S1' and S2' were
  searched against. The baseline gets it densely -- a corner label at all 1,024
  positions of each training screen, and `(w, h)` at every widget corner -- which
  is *more* than S1' consumed (it subsampled 120 positions per image). A
  budget-matched arm at 120 positions per image is reported beside it.
* **Splits.** Identical to `rung3_root.py`: train seeds 0-5 `split='train'`,
  validation seeds 50-52 `split='validation'`, held-out seeds 200-211
  `split='test'` -- the same 12 screens and 227 widgets §33 reports. Model
  selection uses the validation screens only.
* **Trivial reference.** `parent = root`, `no corner anywhere`, and the empty
  parse, computed on the same 12 screens.

Run: `.venv/bin/python research/neural-baselines/visual_baseline.py`
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
# NOTE: `research/visual-ladder/` is deliberately NOT put on sys.path -- it also
# has a `common.py`, and putting it first would silently shadow this directory's
# harness. The track's modules are loaded by file path below instead.

import torch
import torch.nn as nn

import common as harness

# the visual track's own modules; `common` there is shadowed by ours, so its
# helpers are reached through the modules that import it internally.
import importlib.util


def _track(name):
    spec = importlib.util.spec_from_file_location(
        f"vl_{name}", ROOT / "research" / "visual-ladder" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


torch.set_num_threads(1)

VL = _track("common")
sys.modules["common"] = VL           # rung3_widgets / rekey / rung3_root import `common`
R = _track("rung3_widgets")
sys.modules["rung3_widgets"] = R
_track("rekey")
ROOTRULE = _track("rung3_root")
sys.modules["common"] = harness      # hand `common` back to us

FLAT = VL.FLAT
TRAIN_SEEDS = range(0, 6)
VAL_SEEDS = range(50, 53)
TEST_FIRST, TEST_SCREENS = 200, 12
MAXWH = 32


# --- data ------------------------------------------------------------------
def screen(seed, split):
    ep = VL.episode(seed, split, **FLAT)
    w, h = ep["width"], ep["height"]
    px = torch.tensor(ep["pixels"], dtype=torch.float32).reshape(h, w, 3).permute(2, 0, 1) / 255.
    corner = torch.zeros(h, w)
    wt = torch.zeros(h, w, dtype=torch.long)
    ht = torch.zeros(h, w, dtype=torch.long)
    for d in ep["probes"]["hierarchy"]:
        x, y, ww, hh = d["rect"]
        corner[y, x] = 1.
        wt[y, x] = min(ww, MAXWH)
        ht[y, x] = min(hh, MAXWH)
    return {"ep": ep, "pixels": px, "corner": corner, "w": wt, "h": ht,
            "width": w, "height": h, "seed": seed, "split": split}


def stack(screens):
    return {"pixels": torch.stack([s["pixels"] for s in screens]),
            "corner": torch.stack([s["corner"] for s in screens]),
            "w": torch.stack([s["w"] for s in screens]),
            "h": torch.stack([s["h"] for s in screens])}


def subsample_mask(screens, per_image, seed=4):
    """The 120-positions-per-image budget S1' actually consumed."""
    import random
    rng = random.Random(seed)
    masks = []
    for s in screens:
        n = s["height"] * s["width"]
        keep = sorted(set(rng.sample(range(n), min(per_image, n))) | {0})
        m = torch.zeros(n)
        m[torch.tensor(keep)] = 1.
        masks.append(m.reshape(s["height"], s["width"]))
    return torch.stack(masks)


# --- model -----------------------------------------------------------------
class ParseCNN(nn.Module):
    """Dilated per-pixel CNN. The dilation ladder 1,2,4,8,16 gives a 63x63
    receptive field, which is what the extent heads need: a widget run can be up
    to 32 pixels long, so a plain 3x3 stack cannot see the end of one."""

    def __init__(self, width=32, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        layers, chans = [], 3
        for d in dilations:
            layers += [nn.Conv2d(chans, width, 3, padding=d, dilation=d), nn.ReLU()]
            chans = width
        self.body = nn.Sequential(*layers)
        self.corner = nn.Conv2d(chans, 1, 1)
        self.w = nn.Conv2d(chans, MAXWH + 1, 1)
        self.h = nn.Conv2d(chans, MAXWH + 1, 1)

    def forward(self, x):
        f = self.body(x)
        return self.corner(f).squeeze(1), self.w(f), self.h(f)


ZOO = {
    "cnn_w16_d5": lambda: ParseCNN(16, (1, 2, 4, 8, 16)),
    "cnn_w32_d5": lambda: ParseCNN(32, (1, 2, 4, 8, 16)),
    "cnn_w32_d7": lambda: ParseCNN(32, (1, 2, 4, 8, 16, 1, 1)),
    "cnn_w64_d5": lambda: ParseCNN(64, (1, 2, 4, 8, 16)),
    "cnn_w32_rf17": lambda: ParseCNN(32, (1, 1, 1, 1, 1, 1, 1, 1)),
}


def train_one(name, data, mask=None, steps=3000, lr=3e-3, seed=0, pos_weight=40.0, batch=6):
    torch.manual_seed(seed)
    model = ZOO[name]()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight), reduction="none")
    ce = nn.CrossEntropyLoss(reduction="none")
    m = torch.ones_like(data["corner"]) if mask is None else mask
    n = data["pixels"].shape[0]
    generator = torch.Generator().manual_seed(seed)
    started = time.perf_counter()
    for _ in range(steps):
        idx = (torch.arange(n) if n <= batch
               else torch.randint(0, n, (batch,), generator=generator))
        opt.zero_grad()
        c, wl, hl = model(data["pixels"][idx])
        mi = m[idx]
        loss = (bce(c, data["corner"][idx]) * mi).sum() / mi.sum().clamp(min=1)
        at = data["corner"][idx] > 0
        if at.any():
            loss = loss + (ce(wl, data["w"][idx])[at].mean() + ce(hl, data["h"][idx])[at].mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    return model, {"train_seconds": time.perf_counter() - started,
                   "final_loss": float(loss.detach()),
                   "steps": steps, "lr": lr, "batch": min(batch, n), "train_screens": n,
                   "screens_consumed": steps * min(batch, n),
                   "parameters": sum(p.numel() for p in model.parameters())}


# --- prediction and scoring ------------------------------------------------
@torch.no_grad()
def parse(model, s, threshold=0.0):
    c, wl, hl = model(s["pixels"].unsqueeze(0))
    keep = (c[0] > threshold).nonzero()
    ws = wl[0].argmax(0)
    hs = hl[0].argmax(0)
    ep = s["ep"]
    rows = []
    for y, x in keep.tolist():
        own = R.pack_rgb(VL.colour_at(ep, x, y))
        par = own if x == 0 else R.pack_rgb(VL.colour_at(ep, x - 1, y))
        rows.append([x, y, max(1, int(ws[y, x])), max(1, int(hs[y, x])), own, par])
    return sorted(rows)


THRESHOLDS = (-4., -3., -2., -1., -0.5, 0., 0.5, 1., 2., 3., 4., 5., 6.)


def evaluate(model, screens, threshold=0.0):
    rows, exact = [], 0
    corner_tp = corner_fp = corner_fn = 0
    extent_right = extent_total = 0
    for s in screens:
        predicted = parse(model, s, threshold)
        rows.append(ROOTRULE.score_with_root(predicted, s["ep"]) | {"seed": s["seed"]})
        want = {tuple(d["rect"]) for d in s["ep"]["probes"]["hierarchy"]}
        got = {tuple(r[:4]) for r in predicted}
        exact += len(want & got)
        # the two heads, separated, so a failure can be attributed
        want_c = {(d["rect"][0], d["rect"][1]) for d in s["ep"]["probes"]["hierarchy"]}
        got_c = {(r[0], r[1]) for r in predicted}
        corner_tp += len(want_c & got_c)
        corner_fp += len(got_c - want_c)
        corner_fn += len(want_c - got_c)
        sizes = {(d["rect"][0], d["rect"][1]): (d["rect"][2], d["rect"][3])
                 for d in s["ep"]["probes"]["hierarchy"]}
        for r in predicted:
            if (r[0], r[1]) in sizes:
                extent_total += 1
                extent_right += sizes[(r[0], r[1])] == (r[2], r[3])
    total = {"screens": len(rows), "threshold": threshold,
             "widgets_in_probe": sum(r["widgets_in_probe"] for r in rows),
             "rects_predicted": sum(r["rects_predicted"] for r in rows),
             "screens_rects_exact": sum(bool(r["rects_exact"]) for r in rows),
             "roots_predicted": sum(r["roots_predicted"] for r in rows),
             "parent_links_correct": sum(r["parent_links_correct"] for r in rows),
             "parent_links_wrong": sum(r["parent_links_wrong"] for r in rows),
             "trees_exact": sum(bool(r["tree_exact"]) for r in rows),
             "corner_true_positives": corner_tp, "corner_false_positives": corner_fp,
             "corner_false_negatives": corner_fn,
             "corner_recall": corner_tp / max(1, corner_tp + corner_fn),
             "corner_precision": corner_tp / max(1, corner_tp + corner_fp),
             "extent_exact_at_true_corners": extent_right,
             "extent_scored": extent_total,
             "extent_accuracy": extent_right / max(1, extent_total)}
    total["rects_exactly_right"] = exact
    total["rect_accuracy"] = exact / max(1, total["widgets_in_probe"])
    total["link_accuracy"] = total["parent_links_correct"] / max(1, total["widgets_in_probe"])
    return {"episodes": rows, "totals": total}


def choose_threshold(model, validation):
    """Selected on the 3 validation screens only. Never on the held-out 12."""
    best, best_score = 0.0, -1.0
    for t in THRESHOLDS:
        score = evaluate(model, validation, t)["totals"]["link_accuracy"]
        if score > best_score:
            best, best_score = t, score
    return best, best_score


def trivial_references(screens):
    """`parent = root`, `no corner`, and the empty parse, on the same screens."""
    non_root = parent_is_root = widgets = positions = 0
    for s in screens:
        hierarchy = s["ep"]["probes"]["hierarchy"]
        root = [d["id"] for d in hierarchy if d["parent"] == 255][0]
        non_root += sum(d["parent"] != 255 for d in hierarchy)
        parent_is_root += sum(d["parent"] == root for d in hierarchy if d["parent"] != 255)
        widgets += len(hierarchy)
        positions += s["width"] * s["height"]
    return {"widgets": widgets, "positions": positions,
            "parent_is_root_link_accuracy": parent_is_root / max(1, non_root),
            "no_corner_position_accuracy": 1 - widgets / max(1, positions),
            "empty_parse_rect_accuracy": 0.0, "empty_parse_trees_exact": 0}


# --- main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--over-screens", type=int, nargs="*", default=[48, 192])
    ap.add_argument("--models", nargs="*", default=list(ZOO))
    ap.add_argument("--budgets", nargs="*", default=None)
    ap.add_argument("--tag", default="visual")
    args = ap.parse_args()

    print("rendering screens ...", flush=True)
    train = [screen(s, "train") for s in TRAIN_SEEDS]
    val = [screen(s, "validation") for s in VAL_SEEDS]
    test = [screen(TEST_FIRST + i, "test") for i in range(TEST_SCREENS)]
    biggest = [screen(s, "train") for s in range(max(args.over_screens or [0]) or 0)]
    data = {"matched_6_screens": stack(train), "s1_budget_120_positions": stack(train)}
    masks = {"matched_6_screens": None,
             "s1_budget_120_positions": subsample_mask(train, 120)}
    for k in args.over_screens:
        data[f"over_budget_{k}_screens"] = stack(biggest[:k])
        masks[f"over_budget_{k}_screens"] = None

    report = {"configuration": FLAT,
              "splits": {"train_seeds": list(TRAIN_SEEDS), "validation_seeds": list(VAL_SEEDS),
                         "test_seeds": [TEST_FIRST + i for i in range(TEST_SCREENS)],
                         "over_budget_train_screens": args.over_screens},
              "trivial_reference_test": trivial_references(test),
              "trivial_reference_train": trivial_references(train),
              "arms": []}
    print(json.dumps(report["trivial_reference_test"], indent=1), flush=True)

    budgets = args.budgets or list(data)
    for budget in budgets:
        for name in args.models:
            rows = []
            for seed in range(args.seeds):
                model, info = train_one(name, data[budget], masks[budget],
                                        steps=args.steps, seed=seed)
                threshold, val_score = choose_threshold(model, val)
                tr = evaluate(model, train, threshold)["totals"]
                va = evaluate(model, val, threshold)["totals"]
                te = evaluate(model, test, threshold)
                row = {"seed": seed, **info, "threshold": threshold,
                       "train_link_accuracy": tr["link_accuracy"],
                       "train_trees_exact": tr["trees_exact"],
                       "val_link_accuracy": va["link_accuracy"],
                       "val_trees_exact": va["trees_exact"],
                       "test": te["totals"],
                       "test_at_threshold_zero": evaluate(model, test, 0.0)["totals"]}
                rows.append(row)
                t = te["totals"]
                print(f"  {budget:24s} {name:14s} seed {seed}: params {info['parameters']} "
                      f"thr {threshold:+.1f} train links {tr['link_accuracy']:.3f} "
                      f"| test rects {t['rects_exactly_right']}/{t['widgets_in_probe']} "
                      f"links {t['parent_links_correct']}/{t['widgets_in_probe']} "
                      f"trees {t['trees_exact']}/{t['screens']} "
                      f"corner P{t['corner_precision']:.2f}/R{t['corner_recall']:.2f} "
                      f"extent {t['extent_accuracy']:.2f} "
                      f"({info['train_seconds']:.0f}s)", flush=True)
            report["arms"].append({
                "budget": budget, "model": name, "parameters": rows[0]["parameters"],
                "val_link_accuracy_median": statistics.median(r["val_link_accuracy"] for r in rows),
                "test_link_accuracy_median": statistics.median(
                    r["test"]["link_accuracy"] for r in rows),
                "test_rect_accuracy_median": statistics.median(
                    r["test"]["rect_accuracy"] for r in rows),
                "test_trees_exact_median": statistics.median(
                    r["test"]["trees_exact"] for r in rows),
                "train_seconds_median": statistics.median(r["train_seconds"] for r in rows),
                "rows": rows})
            harness.dump(args.tag, report)

    keys = ("budget", "model", "parameters", "val_link_accuracy_median",
            "test_link_accuracy_median", "test_rect_accuracy_median",
            "test_trees_exact_median", "train_seconds_median")
    selected = {}
    for budget in budgets:
        arms = [a for a in report["arms"] if a["budget"] == budget]
        best = max(arms, key=lambda a: (a["val_link_accuracy_median"], -a["parameters"]))
        selected[budget] = {k: best[k] for k in keys}
    report["selected_on_validation"] = selected
    harness.dump(args.tag, report)
    print(json.dumps(selected, indent=1))


if __name__ == "__main__":
    main()
