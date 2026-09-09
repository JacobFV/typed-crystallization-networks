"""What is recoverable from the visual arm's context, before any model is trained.

This is the repository's own recoverability method (`research/object-identity`,
`research/gui-hierarchy`, `research/visual-ladder/common.py:lookup_bound`): a
lookup table keyed by a context is an upper bound on *every* function of that
context, so it says whether a failure is an information failure or a fitting
failure. It is the check that makes "the CNN did badly" interpretable.

Three contexts, all over the corner label at every one of the 1,024 positions:

* `raw_3x3`      -- the 27 raw bytes of the 3x3 window. What a small CNN sees.
* `equality_3`   -- three bits: same-colour-as-left, as-up, as-up-left. DECLARED
                    STRUCTURAL HINT: it is the reduction S0 supplies, and it is
                    reported to show *where* the information sits, not as a
                    baseline the CNN should be measured against.
* `plain_python` -- the exact rule written out, scored on the same screens, as
                    the ceiling both methods are chasing.

`oracle` fits on the evaluation screens themselves (a hard ceiling); `transfer`
fits on the 6 training screens and reads out on the 12 held-out ones (what a
learned table achieves).

Run: `.venv/bin/python research/neural-baselines/visual_bounds.py`
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import common as harness
import visual_baseline as VB          # loads the visual track's modules by path

VL = VB.VL


def contexts(ep, kind):
    w, h = ep["width"], ep["height"]
    corners = {(d["rect"][0], d["rect"][1]) for d in ep["probes"]["hierarchy"]}
    rows = []
    for y in range(h):
        for x in range(w):
            here = VL.colour_at(ep, x, y)
            if kind == "raw_3x3":
                key = tuple(VL.colour_at(ep, min(max(x + dx, 0), w - 1),
                                         min(max(y + dy, 0), h - 1))
                            for dy in (-1, 0, 1) for dx in (-1, 0, 1))
            else:
                left = VL.colour_at(ep, x - 1, y) if x > 0 else None
                up = VL.colour_at(ep, x, y - 1) if y > 0 else None
                upleft = VL.colour_at(ep, x - 1, y - 1) if x > 0 and y > 0 else None
                key = (here == left, here == up, here == upleft)
            rows.append((key, (x, y) in corners))
    return rows


def plain_python_rule(ep):
    """S1' written out: a corner is a pixel differing from its left and its upper
    neighbour, with an off-screen neighbour counting as different."""
    w, h = ep["width"], ep["height"]
    got = set()
    for y in range(h):
        for x in range(w):
            here = VL.colour_at(ep, x, y)
            left = VL.colour_at(ep, x - 1, y) if x > 0 else None
            up = VL.colour_at(ep, x, y - 1) if y > 0 else None
            if here != left and here != up:
                got.add((x, y))
    want = {(d["rect"][0], d["rect"][1]) for d in ep["probes"]["hierarchy"]}
    return {"true_positives": len(got & want), "false_positives": len(got - want),
            "false_negatives": len(want - got), "positions": w * h}


def main():
    train = [VL.episode(s, "train", **VB.FLAT) for s in VB.TRAIN_SEEDS]
    test = [VL.episode(VB.TEST_FIRST + i, "test", **VB.FLAT)
            for i in range(VB.TEST_SCREENS)]

    report = {"configuration": VB.FLAT, "bounds": {}}
    for kind in ("raw_3x3", "equality_3"):
        fit = [row for ep in train for row in contexts(ep, kind)]
        evaluation = [row for ep in test for row in contexts(ep, kind)]
        report["bounds"][kind] = VL.lookup_bound(fit, evaluation)
        harness.report(f"corner lookup bound / {kind}",
                       json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                   for k, v in report["bounds"][kind].items()}))

    totals = {"true_positives": 0, "false_positives": 0, "false_negatives": 0,
              "positions": 0}
    for ep in test:
        row = plain_python_rule(ep)
        for k in totals:
            totals[k] += row[k]
    report["plain_python_rule_on_heldout"] = totals
    harness.report("plain-python corner rule on the 12 held-out screens",
                   json.dumps(totals))
    harness.dump("visual_bounds", report)


if __name__ == "__main__":
    main()
