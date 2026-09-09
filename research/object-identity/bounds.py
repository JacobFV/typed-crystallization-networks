"""Information bounds for object identity, computed BEFORE any search.

Three parts, in the order they should be read.

0.  Two *structural* certificates about the renderer, not statistics: the image
    is invariant to permuting the generator's object list while `object_ids` is
    not, and it is invariant to nothing about the ids while the colours that
    carry them are re-drawn every episode.  Together they say `object_ids` is
    not a function of the observation *at any context size, up to and including
    the entire image*, so no context bound can be above baseline and no search
    over any context can succeed.
1.  The context x target ceiling table: the best possible predictor of each
    target from exactly each context, fitted as a lookup table (outside the
    algebra on purpose -- it upper bounds every program that reads only that
    context and fits the training records).
2.  The obstruction statistics that explain the table.

Run: .venv/bin/python research/object-identity/bounds.py
"""
from __future__ import annotations
import collections, copy, itertools, json, time
import numpy as np

import common2 as C
from common2 import OUT
import common as DP                                   # discrete-perception


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(obj, indent=1, default=str))
    print("wrote", p, flush=True)
    return p
from tcn.generation import Host, Value
from generators.geometry.render import render

R = 8
OBJECTS = 6
TRAIN = range(0, 12)
HELD = range(100, 112)
DELTA = (-2.4, -2.4, -4.2)


# ------------------------------------------------------------------ part 0
def permutation_certificate(seeds=range(0, 8)):
    """Permute the generator's object list; the image is unchanged, the ids are not.

    `render` is called here as an AUDIT oracle on generator state, never as an
    agent input.  It is the generator's own renderer, unmodified.
    """
    rows = []
    for seed in seeds:
        h = Host.create("geometry", seed=seed, split="train",
                        configuration={"resolution": R, "objects": OBJECTS, "horizon": 4})
        h.step([DP.approach(DELTA)])
        s = h.state
        rgb0, d0, ids0, n0 = render(s["objects"], s["camera"], R, R)
        perm = list(range(len(s["objects"])))[::-1]
        objs = [copy.deepcopy(s["objects"][i]) for i in perm]
        rgb1, d1, ids1, n1 = render(objs, s["camera"], R, R)
        inv = {old: new for new, old in enumerate(perm)}
        remapped = np.array([[inv[int(v)] if v >= 0 else -1 for v in row] for row in ids0])
        rows.append({
            "seed": seed,
            "rgb_identical": bool(np.array_equal(rgb0, rgb1)),
            "depth_identical": bool(np.allclose(d0, d1)),
            "ids_identical": bool(np.array_equal(ids0, ids1)),
            "ids_agree_after_remap": bool(np.array_equal(remapped, ids1)),
            "ids_changed_pixels": int((ids0 != ids1).sum()),
            "foreground_pixels": int((ids0 >= 0).sum()),
        })
    return rows


def colour_certificate(seeds=range(0, 8)):
    """Re-draw every object's colour: the ids are identical, the image is not.

    The converse direction of the same statement -- the observation carries the
    colours and the ids carry the list order, and the two are independent.
    """
    rng = np.random.default_rng(7)
    rows = []
    for seed in seeds:
        h = Host.create("geometry", seed=seed, split="train",
                        configuration={"resolution": R, "objects": OBJECTS, "horizon": 4})
        h.step([DP.approach(DELTA)])
        s = h.state
        rgb0, _, ids0, _ = render(s["objects"], s["camera"], R, R)
        objs = copy.deepcopy(s["objects"])
        for o in objs:
            o["color"] = [int(x) for x in rng.integers(40, 240, 3)]
        rgb1, _, ids1, _ = render(objs, s["camera"], R, R)
        rows.append({"seed": seed,
                     "ids_identical": bool(np.array_equal(ids0, ids1)),
                     "rgb_identical": bool(np.array_equal(rgb0, rgb1)),
                     "changed_pixels": int((rgb0 != rgb1).any(-1).sum())})
    return rows


def colour_reuse(train_eps):
    """How often does a foreground colour recur across episodes at all?"""
    per = []
    for pixels, probes in train_eps:
        g = C.rgb_grid(pixels, R)
        per.append({p for row in g for p in row if p != C.BACKGROUND})
    inter = collections.Counter()
    for s in per:
        for k in s:
            inter[k] += 1
    return {"episodes": len(per), "distinct_fg_colours": len(inter),
            "colours_in_more_than_one_episode": sum(1 for v in inter.values() if v > 1),
            "reuse_fraction": sum(1 for v in inter.values() if v > 1) / max(1, len(inter))}


def obstruction_stats(eps):
    per_obj, single = [], 0
    adj_same_diff_colour = adj_same = adj_diff = adj_diff_diff_colour = 0
    for pixels, probes in eps:
        g = C.rgb_grid(pixels, R); ids = C.ids_grid(probes, R)
        by = collections.defaultdict(set)
        for r in range(R):
            for c in range(R):
                if ids[r][c] >= 0:
                    by[ids[r][c]].add(g[r][c])
        for k, v in by.items():
            per_obj.append(len(v)); single += (len(v) == 1)
        for r in range(R):
            for c in range(R - 1):
                a, b = ids[r][c], ids[r][c + 1]
                if a == b:
                    adj_same += 1; adj_same_diff_colour += (g[r][c] != g[r][c + 1])
                else:
                    adj_diff += 1; adj_diff_diff_colour += (g[r][c] != g[r][c + 1])
    return {"object_instances": len(per_obj),
            "mean_distinct_rgb_per_object": sum(per_obj) / max(1, len(per_obj)),
            "max_distinct_rgb_per_object": max(per_obj, default=0),
            "single_coloured_fraction": single / max(1, len(per_obj)),
            "adjacent_same_object_pairs": adj_same,
            "adjacent_same_object_different_colour": adj_same_diff_colour,
            "adjacent_same_object_different_colour_fraction": adj_same_diff_colour / max(1, adj_same),
            "adjacent_different_object_pairs": adj_diff,
            "adjacent_different_object_different_colour": adj_diff_diff_colour,
            "adjacent_different_object_same_colour_fraction":
                (adj_diff - adj_diff_diff_colour) / max(1, adj_diff)}


CONTEXTS = ["pixel", "pixel_pos", "pair_right", "cross4", "win3x3", "pixel_agg", "agg_only",
            "eq_pair_right", "eq_cross4", "eq_win3x3", "eqbg_cross4", "eqbg_win3x3",
            "bg_pair_right", "chroma_pair_right", "chroma_cross4"]
TARGETS = ["fg", "object_ids", "is_object_0", "raster_rank", "same_right", "same_down",
           "fg_edge", "obj_edge_fg", "near",
           "object_ids_fg", "raster_rank_fg", "near_fg", "shade_bin_fg", "normal_up_fg"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    train = C.load(TRAIN, R, OBJECTS, "train")
    held = C.load(HELD, R, OBJECTS, "train")
    DP.report("episodes loaded", f"{len(train)} train / {len(held)} held, {time.perf_counter()-t0:.1f}s")

    perm = permutation_certificate()
    col = colour_certificate()
    reuse = colour_reuse(train + held)
    obstruction = obstruction_stats(train)
    DP.report("permutation: rgb identical in", f"{sum(r['rgb_identical'] for r in perm)}/{len(perm)}")
    DP.report("permutation: ids identical in", f"{sum(r['ids_identical'] for r in perm)}/{len(perm)}")
    DP.report("colour redraw: ids identical in", f"{sum(r['ids_identical'] for r in col)}/{len(col)}")

    table = []
    for ctx in CONTEXTS:
        for tgt in TARGETS:
            row = C.table_ceiling(train, held, R, ctx, tgt)
            table.append(row)
            print(f"  {ctx:16s} {tgt:14s} train={row['train_acc']:.4f} "
                  f"held={row['held_acc']:.4f} base={row['majority_baseline']:.4f} "
                  f"adv={row['advantage']:+.4f} keys={row['keys']:6d} "
                  f"recur={row['held_key_recurrence']:.3f}", flush=True)

    dump("bounds", {"resolution": R, "objects": OBJECTS,
                       "train_seeds": list(TRAIN), "held_seeds": list(HELD),
                       "permutation_certificate": perm,
                       "colour_certificate": col,
                       "colour_reuse": reuse,
                       "obstruction": obstruction,
                       "ceilings": table,
                       "seconds": time.perf_counter() - t0})


if __name__ == "__main__":
    main()
