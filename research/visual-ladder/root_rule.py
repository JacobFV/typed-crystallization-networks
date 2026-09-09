"""Is the root recoverable as a program?  The rule check that decides it.

H6 excluded position (0, 0): the corner predicate reads a left and an upper
neighbour, so the root is outside the position set by construction and is
supplied to the scorer.  The proposed repair is to treat an off-screen neighbour
as "different", which makes (0, 0) a corner.  Before searching for that program,
check the rule at every instance, the way section 1.4 checks the others.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import FLAT, colour_at, dump, episode, report


def run(seeds, split, **configuration):
    out = {"tp": 0, "fp": 0, "fn": 0, "positions": 0, "widgets": 0,
           "extent_exact": 0, "extent_checked": 0,
           "root_extent_exact": 0, "root_extent_checked": 0,
           "non_root_at_edge": 0, "fp_detail": [], "unclamped_positions": 0}
    for s in seeds:
        ep = episode(s, split, **configuration)
        W, H = ep["width"], ep["height"]
        px = lambda x, y: colour_at(ep, x, y)
        hierarchy = ep["probes"]["hierarchy"]
        corners = {(d["rect"][0], d["rect"][1]): d for d in hierarchy}
        out["widgets"] += len(hierarchy)
        out["non_root_at_edge"] += sum(1 for d in hierarchy
                                       if d["parent"] != 255
                                       and (d["rect"][0] == 0 or d["rect"][1] == 0))
        out["unclamped_positions"] += (W - 1) * (H - 1)
        for y in range(H):
            for x in range(W):
                out["positions"] += 1
                # off-screen neighbour counts as "different"
                left_diff = x == 0 or px(x - 1, y) != px(x, y)
                up_diff = y == 0 or px(x, y - 1) != px(x, y)
                predicted = left_diff and up_diff
                if predicted and (x, y) in corners:
                    out["tp"] += 1
                elif predicted:
                    out["fp"] += 1
                    if len(out["fp_detail"]) < 20:
                        out["fp_detail"].append([s, x, y])
                elif (x, y) in corners:
                    out["fn"] += 1
        # the extent rule, as a contiguous same-colour run, at every corner
        for (x, y), d in corners.items():
            w = 1
            while x + w < W and px(x + w, y) == px(x, y):
                w += 1
            h = 1
            while y + h < H and px(x, y + h) == px(x, y):
                h += 1
            ok = (w, h) == (d["rect"][2], d["rect"][3])
            out["extent_checked"] += 1
            out["extent_exact"] += ok
            if d["parent"] == 255:
                out["root_extent_checked"] += 1
                out["root_extent_exact"] += ok
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=200)
    ap.add_argument("--screens", type=int, default=12)
    ap.add_argument("--tag", default="root_rule")
    args = ap.parse_args()
    configuration = dict(FLAT)
    held = run(range(args.first, args.first + args.screens), "test", **configuration)
    train = run(range(6), "train", **configuration)
    report("held-out (seeds 200-211)", held)
    report("train (seeds 0-5)", train)
    dump(args.tag, {"held": held, "train": train, "configuration": configuration,
                    "arguments": vars(args)})


if __name__ == "__main__":
    main()
