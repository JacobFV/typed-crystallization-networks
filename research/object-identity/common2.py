"""Shared data helpers for the object-identity track.

Reuses `research/discrete-perception/common.py` verbatim (imported, not copied)
for the episode loader, the graph builder, the three-node positional caller,
the enumeration wrappers and `local_fit`.  Only what is new to this track lives
here: context extraction, the lookup-table ceiling estimator, and the
permutation/relabelling certificates that call the generator's own renderer as
an *audit* oracle (never as an agent input).
"""
from __future__ import annotations
import sys, pathlib, json, itertools, collections, copy
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "research" / "discrete-perception"))
sys.path.insert(0, str(ROOT))
import common                                  # noqa: E402  (discrete-perception)

OUT = HERE / "out"
BACKGROUND = (24, 30, 43)
SENTINEL = (256, 256, 256)   # off-alphabet border marker, never a real byte


def load(seeds, resolution=8, objects=6, split="train"):
    """[(pixels, probes)] with the declared `camera` approach action."""
    return [common.episode(s, resolution, objects=objects, split=split) for s in seeds]


def rgb_grid(pixels, R):
    return [[tuple(int(pixels[3 * (r * R + c) + k]) for k in range(3)) for c in range(R)]
            for r in range(R)]


def ids_grid(probes, R):
    v = probes["object_ids"]
    return [[int(v[r * R + c]) for c in range(R)] for r in range(R)]


def depth_grid(probes, R):
    v = probes["depth"]
    return [[float(v[r * R + c]) for c in range(R)] for r in range(R)]


def at(grid, r, c, R, default=SENTINEL):
    return grid[r][c] if 0 <= r < R and 0 <= c < R else default


# ---------------------------------------------------------------- contexts
def colour_rank(rgbs, R):
    """Rank of each pixel's colour among distinct colours by first raster appearance.

    A pixel plus one global aggregate of the whole image, and colour-invariant:
    re-drawing every object's colour permutes nothing here as long as raster
    order is preserved.
    """
    order, seen = {}, 0
    for r in range(R):
        for c in range(R):
            k = rgbs[r][c]
            if k not in order:
                order[k] = seen; seen += 1
    return [[order[rgbs[r][c]] for c in range(R)] for r in range(R)]


def contexts(rgbs, R, ranks):
    """name -> function(r, c) -> hashable key."""
    def P(r, c): return at(rgbs, r, c, R)
    return {
        "pixel":          lambda r, c: P(r, c),
        "pixel_pos":      lambda r, c: (P(r, c), r, c),
        "pair_right":     lambda r, c: (P(r, c), P(r, c + 1)),
        "cross4":         lambda r, c: (P(r, c), P(r - 1, c), P(r + 1, c), P(r, c - 1), P(r, c + 1)),
        "win3x3":         lambda r, c: tuple(P(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)),
        "pixel_agg":      lambda r, c: (P(r, c), ranks[r][c]),
        "agg_only":       lambda r, c: (ranks[r][c],),
        # colour-invariant abstractions of the same windows: the only thing the
        # algebra can do to two bytes it has no constant for is compare them.
        "eq_pair_right":  lambda r, c: (P(r, c) == P(r, c + 1),),
        "eq_cross4":      lambda r, c: tuple(P(r, c) == P(r + dr, c + dc)
                                             for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))),
        "eq_win3x3":      lambda r, c: tuple(P(r, c) == P(r + dr, c + dc)
                                             for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                                             if (dr, dc) != (0, 0)),
        # equality pattern plus the one thing a byte constant buys: is it background
        "eqbg_cross4":    lambda r, c: (P(r, c) == BACKGROUND,) + tuple(
                                             P(r, c) == P(r + dr, c + dc)
                                             for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))),
        "eqbg_win3x3":    lambda r, c: (P(r, c) == BACKGROUND,) + tuple(
                                             P(r, c) == P(r + dr, c + dc)
                                             for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                                             if (dr, dc) != (0, 0)),
        "bg_pair_right":  lambda r, c: (P(r, c) == BACKGROUND, P(r, c + 1) == BACKGROUND,
                                        P(r, c) == P(r, c + 1)),
        # chromaticity: NOT expressible on `role="byte"` pixels (they are
        # non-numeric, so no ratio, no division, no ordering).  Measured as a
        # bound on what a hypothetical arithmetic-on-bytes family could reach.
        "chroma_pair_right": lambda r, c: (P(r, c) == BACKGROUND, P(r, c + 1) == BACKGROUND,
                                           same_chroma(P(r, c), P(r, c + 1))),
        "chroma_cross4":  lambda r, c: (P(r, c) == BACKGROUND,) + tuple(
                                           same_chroma(P(r, c), P(r + dr, c + dc))
                                           for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))),
    }


def same_chroma(a, b, tol=.04):
    """Do two pixels lie on the same colour ray through the origin?

    The renderer paints `clip(base_colour * shade)`, so every pixel of one
    object is a positive scalar multiple of that object's base colour and two
    pixels of the same object are collinear whatever their facing.  This is the
    exact invariant the equality abstraction cannot see.
    """
    if a == SENTINEL or b == SENTINEL:
        return None
    na = sum(a); nb = sum(b)
    if na == 0 or nb == 0:
        return a == b
    return all(abs(x / na - y / nb) <= tol for x, y in zip(a, b))


def normals_grid(probes, R):
    v = probes["normals"]
    return [[tuple(float(v[3 * (r * R + c) + k]) for k in range(3)) for c in range(R)]
            for r in range(R)]


def targets(ids, depths, R, ranks, normals=None):
    """name -> function(r, c) -> integer label, or None where undefined."""
    def I(r, c): return ids[r][c] if 0 <= r < R and 0 <= c < R else None
    med = 0.573
    order, seen = {}, 0
    for r in range(R):
        for c in range(R):
            k = ids[r][c]
            if k >= 0 and k not in order:
                order[k] = seen; seen += 1
    def rank(r, c): return order.get(ids[r][c], -1)

    def same_right(r, c):
        b = I(r, c + 1)
        return None if b is None else int(I(r, c) == b)

    def same_down(r, c):
        b = I(r + 1, c)
        return None if b is None else int(I(r, c) == b)

    def fg_edge(r, c):
        b = I(r, c + 1)
        return None if b is None else int((I(r, c) >= 0) != (b >= 0))

    def obj_edge_fg(r, c):
        """boundary between two DIFFERENT objects, both foreground (the new part)."""
        b = I(r, c + 1); a = I(r, c)
        if b is None or a < 0 or b < 0:
            return None
        return int(a != b)

    def fg_only(f):
        return lambda r, c: (None if I(r, c) is None or I(r, c) < 0 else f(r, c))

    LIGHT = (.3, .8, .5)
    ln = sum(x * x for x in LIGHT) ** .5
    def shade(r, c):
        n = normals[r][c]
        return .3 + .7 * abs(sum(a * b for a, b in zip(n, LIGHT)) / ln)
    def shade_bin(r, c):
        return min(2, int((shade(r, c) - .3) / .7 * 3))
    def normal_up(r, c):
        return int(abs(normals[r][c][1]) > .7)

    return {
        "fg":            lambda r, c: int(I(r, c) >= 0),
        "object_ids_fg": fg_only(lambda r, c: I(r, c)),
        "raster_rank_fg": fg_only(rank),
        "near_fg":       fg_only(lambda r, c: int(0 < depths[r][c] < med)),
        "shade_bin_fg":  fg_only(shade_bin),
        "normal_up_fg":  fg_only(normal_up),
        "object_ids":    lambda r, c: I(r, c),
        "is_object_0":   lambda r, c: int(I(r, c) == 0),
        "raster_rank":   rank,
        "same_right":    same_right,
        "same_down":     same_down,
        "fg_edge":       fg_edge,
        "obj_edge_fg":   obj_edge_fg,
        "near":          lambda r, c: int(0 < depths[r][c] < med),
    }


def table_ceiling(train, held, R, context_name, target_name):
    """Best possible predictor of `target` from exactly `context`, fitted on train.

    Fitted as an explicit lookup table -- outside the operator algebra on
    purpose, because it upper-bounds every program whose only input is that
    context and which is required to fit the training records.  Unseen keys fall
    back to the training majority, which is the same fallback the previous
    track's per-pixel ceiling used.
    """
    def rows(eps):
        out = []
        for pixels, probes in eps:
            g = rgb_grid(pixels, R); ids = ids_grid(probes, R); dz = depth_grid(probes, R)
            nm = normals_grid(probes, R)
            ranks = colour_rank(g, R)
            ctx = contexts(g, R, ranks)[context_name]
            tgt = targets(ids, dz, R, ranks, nm)[target_name]
            for r in range(R):
                for c in range(R):
                    y = tgt(r, c)
                    if y is None:
                        continue
                    out.append((ctx(r, c), y))
        return out

    tr, he = rows(train), rows(held)
    counts = collections.defaultdict(collections.Counter)
    overall = collections.Counter()
    for k, y in tr:
        counts[k][y] += 1; overall[y] += 1
    fallback = overall.most_common(1)[0][0] if overall else 0
    table = {k: v.most_common(1)[0][0] for k, v in counts.items()}
    colliding = sum(1 for k, v in counts.items() if len(v) > 1)
    train_acc = sum(table[k] == y for k, y in tr) / max(1, len(tr))
    seen = sum(1 for k, _ in he if k in table)
    held_acc = sum(table.get(k, fallback) == y for k, y in he) / max(1, len(he))
    hc = collections.Counter(y for _, y in he)
    baseline = hc.most_common(1)[0][1] / max(1, len(he)) if hc else 0.
    return {"context": context_name, "target": target_name,
            "keys": len(table), "colliding_keys": colliding,
            "train_records": len(tr), "held_records": len(he),
            "train_acc": train_acc, "held_acc": held_acc,
            "majority_baseline": baseline, "advantage": held_acc - baseline,
            "held_key_recurrence": seen / max(1, len(he)),
            "classes": len(hc)}
