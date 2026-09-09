"""Re-run S3's parse on held-out screens from the frozen selections in `out/rung3.json`.

Nothing is searched here.  `rung3_widgets.py` writes the chosen selections for
S0, S1 and S2; this rebuilds the three modules from them and reports the parse
against the `hierarchy` probe on a wider held-out set, with the trivial baselines
beside it.  It exists so the parse numbers in `RESULTS.md` are regenerated from a
committed JSON rather than transcribed.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import FLAT, Registry, bytes_type, dump, episode, load, report
from tcn.types import Value

import rung3_widgets as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="rung3")
    ap.add_argument("--tag", default="rung3_parse")
    ap.add_argument("--first", type=int, default=200)
    ap.add_argument("--screens", type=int, default=12)
    args = ap.parse_args()

    found = load(args.source)
    registry = Registry()
    configuration = dict(FLAT)
    probe = episode(0, "train", **configuration)
    W, H = probe["width"], probe["height"]
    offsets = R.offset_pool(W)

    p0 = R.same_scaffold(registry, W, H)
    m0 = registry.register_module(p0.harden(found["s0"]["chosen"]))
    p1 = R.corner_scaffold(registry, W, H, m0, offsets)
    m1 = registry.register_module(p1.harden(found["s1"]["chosen"]))
    p2 = R.rect_scaffold(registry, W, H, m0, offsets, )
    m2 = registry.register_module(p2.harden(found["s2"]["chosen"]))

    rows, base = [], {"non_root": 0, "parent_is_root": 0, "corners": 0, "positions": 0}
    for seed in range(args.first, args.first + args.screens):
        ep = episode(seed, "test", **configuration)
        BT = bytes_type(ep["width"], ep["height"])
        parser = R.assembly(registry, BT, R.interior_positions(ep), m1, m2)
        got = parser.run({"observation": Value.of(BT, ep["pixels"])}, registry=registry)[0]
        row = R.score_tree(sorted(got["mapped"].decoded), ep) | {"seed": seed}
        rows.append(row)
        hierarchy = ep["probes"]["hierarchy"]
        root = [d["id"] for d in hierarchy if d["parent"] == 255][0]
        base["non_root"] += sum(d["parent"] != 255 for d in hierarchy)
        base["parent_is_root"] += sum(d["parent"] == root for d in hierarchy if d["parent"] != 255)
        base["corners"] += len(hierarchy)
        base["positions"] += (ep["width"] - 1) * (ep["height"] - 1)
        report(f"parse seed {seed}", f"rects {row['rects_predicted']}/{row['rects_true']} "
               f"exact={row['rects_exact']} links {row['parent_links_correct']}/"
               f"{row['non_root']} key-collisions {row['key_collisions']}")

    total = {"screens": len(rows),
             "widgets_non_root": sum(r["non_root"] for r in rows),
             "rects_predicted": sum(r["rects_predicted"] for r in rows),
             "screens_rects_exact": sum(r["rects_exact"] for r in rows),
             "parent_links_correct": sum(r["parent_links_correct"] for r in rows),
             "parent_links_wrong": sum(r["parent_links_wrong"] for r in rows),
             "key_collisions": sum(r["key_collisions"] for r in rows),
             "trees_exact": sum(r["tree_exact"] for r in rows)}
    total["rect_recall"] = total["rects_predicted"] / max(1, total["widgets_non_root"])
    total["link_accuracy"] = total["parent_links_correct"] / max(1, total["widgets_non_root"])
    baseline = {"parent_is_root_accuracy": base["parent_is_root"] / max(1, base["non_root"]),
                "no_corner_position_accuracy": 1 - base["corners"] / max(1, base["positions"]),
                "empty_parse_rect_recall": 0.0}
    report("totals", total)
    report("baselines", baseline)
    dump(args.tag, {"episodes": rows, "totals": total, "baselines": baseline,
                    "configuration": configuration, "arguments": vars(args),
                    "selections": {k: found[k]["chosen"] for k in ("s0", "s1", "s2")}})


if __name__ == "__main__":
    main()
