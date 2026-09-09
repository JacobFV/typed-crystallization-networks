"""What the geometry probes actually are, as functions of the observation.

Before searching for a program, measure what a program could possibly compute.
Three questions per target, at each resolution:

  * how much of the image is foreground at all (a target that is 98% constant
    makes every accuracy number a report on that constant);
  * is the target a function of the *pixel*?  A shared per-position module that
    reads only its own three bytes cannot do better than the best RGB lookup
    table, so the collision rate is a hard ceiling on that family;
  * does that lookup table transfer to held-out episodes?  `geometry`
    re-randomises object colours every episode, so a table that fits training
    can still be at chance on new episodes.

Supervision only: probes never reach a program input.
"""
from __future__ import annotations

import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import dump, episode, report

RESOLUTIONS = (4, 8, 12, 16, 24, 32)
TRAIN = tuple(range(12))
HELD = tuple(range(100, 112))


def triples(pixels, n):
    return [tuple(pixels[3 * i:3 * i + 3]) for i in range(n)]


def targets(probes, n):
    depth = probes["depth"]
    ids = probes["object_ids"]
    return {"foreground": [ids[i] >= 0 for i in range(n)],
            "object_ids": [int(ids[i]) for i in range(n)],
            "depth_q": [round(depth[i], 3) for i in range(n)]}


def lookup_transfer(train_pairs, held_pairs):
    """Best RGB->label lookup table fit on train, scored on held-out.

    This is the matched-information ceiling for any per-pixel function: no
    program that reads only the pixel can beat it.
    """
    table = collections.defaultdict(collections.Counter)
    for rgb, y in train_pairs:
        table[rgb][y] += 1
    best = {k: c.most_common(1)[0][0] for k, c in table.items()}
    majority = collections.Counter(y for _, y in train_pairs).most_common(1)[0][0]
    seen = hit = maj_hit = 0
    for rgb, y in held_pairs:
        seen += rgb in best
        hit += (best.get(rgb, majority) == y)
        maj_hit += (majority == y)
    n = max(1, len(held_pairs))
    collisions = sum(1 for c in table.values() if len(c) > 1)
    return {"distinct_rgb_train": len(table), "colliding_rgb_train": collisions,
            "collision_fraction": collisions / max(1, len(table)),
            "train_pairs": len(train_pairs),
            "held_coverage": seen / n, "held_accuracy": hit / n,
            "majority_accuracy": maj_hit / n}


def main():
    out = {"train_seeds": list(TRAIN), "held_seeds": list(HELD), "resolutions": list(RESOLUTIONS),
           "camera": "one declared `camera` action, delta=(-2.4,-2.4,-4.2)", "rows": []}
    for R in RESOLUTIONS:
        n = R * R
        tr = {k: [] for k in ("foreground", "object_ids", "depth_q")}
        hd = {k: [] for k in tr}
        fg = tot = 0
        depth_zero_agrees = True
        ids_seen = collections.Counter()
        for seeds, bucket, split in ((TRAIN, tr, "train"), (HELD, hd, "test")):
            for s in seeds:
                pixels, probes = episode(s, R, split=split)
                rgbs = triples(pixels, n)
                t = targets(probes, n)
                depth_zero_agrees &= all((probes["depth"][i] == 0.) == (probes["object_ids"][i] < 0)
                                         for i in range(n))
                for k in bucket:
                    bucket[k].extend(zip(rgbs, t[k]))
                if bucket is tr:
                    fg += sum(t["foreground"]); tot += n
                    ids_seen.update(t["object_ids"])
        row = {"resolution": R, "pixels_per_image": n,
               "foreground_fraction_train": fg / max(1, tot),
               "depth_zero_equals_no_object": bool(depth_zero_agrees),
               "distinct_object_ids_train": sorted(ids_seen)}
        for k in tr:
            row[k] = lookup_transfer(tr[k], hd[k])
        out["rows"].append(row)
        report(f"R={R:2d} foreground fraction", f"{row['foreground_fraction_train']:.3f}")
        for k in ("foreground", "object_ids", "depth_q"):
            m = row[k]
            report(f"   {k:11s} collisions/held-cover/held-acc/majority",
                   f"{m['collision_fraction']:.3f} / {m['held_coverage']:.3f} / "
                   f"{m['held_accuracy']:.4f} / {m['majority_accuracy']:.4f}")
    dump("audit", out)


if __name__ == "__main__":
    main()
