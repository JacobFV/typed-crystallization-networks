"""Re-key the parent lookup on position rather than colour, and re-measure.

`score_tree` resolved a widget's `parent_key` -- the packed RGB of the pixel to
the left of its corner -- through a dict of the parsed widgets' own fill colours.
`bounds.json`'s `colour_injective` field already said that key is not injective at
`palette 32`, and all 42 wrong links in `out/rung3_parse.json` are its collisions.

Nothing in the learned program changes.  The rectangle module already emits
`(x, y, w, h, own_key, parent_key)`, and fields 0 and 1 *are* the widget's
top-left corner: an exact, injective key that the parse already produces.  What
changes is the resolution step in the scorer:

  colour  : parent = the parsed widget whose fill colour equals `parent_key`
  position: parent = the smallest parsed rectangle containing the parent pixel
            `(x - 1, y)`, whose address the corner position already determines

Both are bookkeeping over the parse's own output; neither reads the probe.  The
parse is run once, its rows cached, and every rule scored against the same rows,
so the comparison is on one set of predictions.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import FLAT, OUT, Registry, bytes_type, colour_at, dump, episode, load, report
from tcn.types import Value

import rung3_widgets as R


# --------------------------------------------------------------------------
def parse_rows(source, first, screens, cache):
    """Run the frozen parse on held-out screens; cache the raw predicted rows."""
    path = OUT / f"{cache}.json"
    if path.exists():
        blob = json.loads(path.read_text())
        if blob["first"] == first and blob["screens"] == screens and blob["source"] == source:
            report("cache", f"reusing {path.name}")
            return blob["rows"]
    found = load(source)
    registry = Registry()
    configuration = dict(FLAT)
    probe = episode(0, "train", **configuration)
    W, H = probe["width"], probe["height"]
    offsets = R.offset_pool(W)
    p0 = R.same_scaffold(registry, W, H)
    m0 = registry.register_module(p0.harden(found["s0"]["chosen"]))
    p1 = R.corner_scaffold(registry, W, H, m0, offsets)
    m1 = registry.register_module(p1.harden(found["s1"]["chosen"]))
    p2 = R.rect_scaffold(registry, W, H, m0, offsets)
    m2 = registry.register_module(p2.harden(found["s2"]["chosen"]))
    rows = {}
    for seed in range(first, first + screens):
        ep = episode(seed, "test", **configuration)
        BT = bytes_type(ep["width"], ep["height"])
        parser = R.assembly(registry, BT, R.interior_positions(ep), m1, m2)
        started = time.perf_counter()
        got = parser.run({"observation": Value.of(BT, ep["pixels"])}, registry=registry)[0]
        rows[str(seed)] = [list(map(int, r)) for r in sorted(got["mapped"].decoded)]
        report(f"parse seed {seed}", f"{len(rows[str(seed)])} rows "
                                     f"{time.perf_counter() - started:.1f}s")
    path.write_text(json.dumps({"first": first, "screens": screens, "source": source,
                                "rows": rows}))
    return rows


# --------------------------------------------------------------------------
def score(predicted, ep, rule):
    """Rectangle recall (rule-independent) and parent links under one rule."""
    truth = {d["id"]: d for d in ep["probes"]["hierarchy"]}
    rects = {i: tuple(d["rect"]) for i, d in truth.items()}
    non_root = [d for d in truth.values() if d["parent"] != 255]
    got_rects = {tuple(r[:4]) for r in predicted}
    want_rects = {rects[d["id"]] for d in non_root}
    screen = (0, 0, ep["width"], ep["height"])

    by_colour = {r[4]: r for r in predicted}
    root_colour = R.pack_rgb(colour_at(ep, 0, 0))
    by_corner = {(r[0], r[1]): r for r in predicted}

    def containing(px, py):
        """Smallest parsed rectangle containing a pixel; None when there is none."""
        inside = [r for r in predicted
                  if r[0] <= px < r[0] + r[2] and r[1] <= py < r[1] + r[3]]
        if not inside:
            return None
        return min(inside, key=lambda r: r[2] * r[3])

    right = wrong = 0
    unresolved = 0
    for r in predicted:
        if rule == "colour":
            key = r[5]
            got = (screen if key == root_colour
                   else tuple(by_colour[key][:4]) if key in by_colour else None)
        elif rule == "position":
            owner = containing(r[0] - 1, r[1])
            got = screen if owner is None else tuple(owner[:4])
        else:
            raise ValueError(rule)
        if got is None:
            unresolved += 1
        match = [d for d in truth.values() if tuple(d["rect"]) == tuple(r[:4])]
        if len(match) == 1 and got is not None and got == tuple(rects[match[0]["parent"]]):
            right += 1
        else:
            wrong += 1
    return {"non_root": len(non_root), "widgets_in_probe": len(truth),
            "rects_predicted": len(got_rects), "rects_true": len(want_rects),
            "rects_exact": got_rects == want_rects,
            "colour_key_collisions": len(predicted) - len({r[4] for r in predicted}),
            "corner_key_collisions": len(predicted) - len(by_corner),
            "parent_links_correct": right, "parent_links_wrong": wrong,
            "unresolved": unresolved,
            "tree_exact": got_rects == want_rects and wrong == 0}


def totals(rows):
    keys = ("non_root", "rects_predicted", "parent_links_correct", "parent_links_wrong",
            "colour_key_collisions", "corner_key_collisions", "unresolved")
    out = {k: sum(r[k] for r in rows) for k in keys}
    out["screens"] = len(rows)
    out["screens_rects_exact"] = sum(r["rects_exact"] for r in rows)
    out["trees_exact"] = sum(r["tree_exact"] for r in rows)
    out["link_accuracy"] = out["parent_links_correct"] / max(1, out["non_root"])
    out["rect_recall"] = out["rects_predicted"] / max(1, out["non_root"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="rung3")
    ap.add_argument("--tag", default="rekey")
    ap.add_argument("--cache", default="rekey_rows")
    ap.add_argument("--first", type=int, default=200)
    ap.add_argument("--screens", type=int, default=12)
    args = ap.parse_args()

    raw = parse_rows(args.source, args.first, args.screens, args.cache)
    configuration = dict(FLAT)
    result = {"arguments": vars(args), "configuration": configuration, "rules": {}}
    base = {"non_root": 0, "parent_is_root": 0}
    collision_split = {}
    for rule in ("colour", "position"):
        per = []
        for seed in range(args.first, args.first + args.screens):
            ep = episode(seed, "test", **configuration)
            per.append(score(raw[str(seed)], ep, rule) | {"seed": seed})
            if rule == "colour":
                hierarchy = ep["probes"]["hierarchy"]
                root = [d["id"] for d in hierarchy if d["parent"] == 255][0]
                base["non_root"] += sum(d["parent"] != 255 for d in hierarchy)
                base["parent_is_root"] += sum(d["parent"] == root for d in hierarchy
                                              if d["parent"] != 255)
        result["rules"][rule] = {"episodes": per, "totals": totals(per)}
        report(f"rule {rule}", result["rules"][rule]["totals"])
        clean = [r for r in per if r["colour_key_collisions"] == 0]
        dirty = [r for r in per if r["colour_key_collisions"] > 0]
        collision_split[rule] = {"collision_free_screens": totals(clean) if clean else None,
                                 "colliding_screens": totals(dirty) if dirty else None}
    result["collision_split"] = collision_split
    result["baselines"] = {"parent_is_root_accuracy": base["parent_is_root"]
                           / max(1, base["non_root"]),
                           "empty_parse_rect_recall": 0.0}
    report("baselines", result["baselines"])
    dump(args.tag, result)


if __name__ == "__main__":
    main()
