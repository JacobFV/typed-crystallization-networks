"""Recoverability bounds, computed BEFORE any search.

For each rung this track intends, the question is asked in the form the
object-identity work uses: *given exactly this context, what is the best possible
predictor?*  A lookup table keyed by the context is an upper bound on every
function of that context, so if it does no better than the majority label, no
program over that context can exist and searching one is wasted.  Two numbers per
rung (see `common.lookup_bound`): the `oracle` bound fitted on the evaluation set
itself, which is a hard ceiling, and the `transfer` bound fitted on training
episodes, which is what a learned table achieves.

Rungs asked about here:

  R1  is a widget edge determined by a local pixel neighbourhood?
        context = the ordered pair of RGB triples at (i, i+step)
        target  = owner(i) != owner(i+step)
  R2  is a glyph determined by its bounding box?
        context = the pixels of the emitted glyph box (raw, and ink-binarised)
        target  = the character code
  R3  is a widget's kind determined by its rendered appearance?
        context = fill colour / (w, h) / both
        target  = kind
  R4  is a widget's parent determined by the geometry alone?
        the smallest strictly-containing rectangle, checked against the truth
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import colour_at, dump, edge_positions, episode, lookup_bound, report
from generators.gui.render import KINDS

TRAIN = tuple(range(16))
EVAL = tuple(range(100, 116))

SETTINGS = {
    "flat (default)": {},
    "borders": {"borders": True},
    "labels": {"resolution": 32, "labels": True},
    "borders+labels": {"resolution": 32, "borders": True, "labels": True},
    "palette 8, 6 widgets": {"palette": 8},
    "palette 4, 12 widgets": {"palette": 4, "widgets": 12, "resolution": 32},
    "colour_mode=kind": {"colour_mode": "kind"},
    "12 widgets, nesting 4": {"resolution": 32, "widgets": 12, "nesting": 4},
    "resolution 32": {"resolution": 32},
}


def edge_records(seeds, split, step_axis="x", pattern=False, **configuration):
    """Context = the raw byte pair, or its per-channel equality pattern.

    The equality pattern is the tighter and more useful bound for this project:
    the only legal predicate on a `role="byte"` value is `eq`, so ANY rung-1
    program over two pixels is a Boolean function of exactly these three bits.
    Its oracle is therefore the ceiling of the whole expressible family, while the
    raw-byte oracle is the ceiling of every function of the neighbourhood
    including ones the algebra cannot write.
    """
    rows = []
    for seed in seeds:
        ep = episode(seed, split, **configuration)
        step = 1 if step_axis == "x" else ep["width"]
        owner = ep["probes"]["owner"]
        for i in edge_positions(ep):
            a, b = colour_at(ep, i), colour_at(ep, i + step)
            context = tuple(x == y for x, y in zip(a, b)) if pattern else (a, b)
            rows.append((context, owner[i] != owner[i + step]))
    return rows


def rung1(pattern=False):
    out = {}
    for name, configuration in SETTINGS.items():
        for axis in ("x", "y"):
            key = f"{name} | {axis}"
            out[key] = lookup_bound(
                edge_records(TRAIN, "train", axis, pattern, **configuration),
                edge_records(EVAL, "test", axis, pattern, **configuration))
            out[key]["configuration"] = configuration
            b = out[key]
            report(key, f"majority {b['majority_baseline']:.4f}  oracle {b['oracle_accuracy']:.4f} "
                        f"(+{b['oracle_advantage']:.4f})  transfer {b['transfer_accuracy']:.4f} "
                        f"(+{b['transfer_advantage']:.4f})  unseen {b['unseen_key_fraction']:.3f}")
    return out


def glyph_records(seeds, split, binarise, **configuration):
    rows = []
    for seed in seeds:
        ep = episode(seed, split, resolution=32, labels=True, **configuration)
        width = ep["width"]
        for g in ep["probes"].get("glyphs", []):
            x, y, w, h = g["rect"]
            patch = tuple(colour_at(ep, (y + dy) * width + (x + dx))
                          for dy in range(h) for dx in range(w))
            if binarise:
                patch = tuple(c == (0, 0, 0) for c in patch)
            rows.append(((w, h, patch), g["code"]))
    return rows


def rung2():
    out = {}
    for binarise in (False, True):
        key = "glyph box, ink-binarised" if binarise else "glyph box, raw bytes"
        out[key] = lookup_bound(glyph_records(TRAIN, "train", binarise),
                                glyph_records(EVAL, "test", binarise))
        b = out[key]
        report(key, f"majority {b['majority_baseline']:.4f}  oracle {b['oracle_accuracy']:.4f} "
                    f"(+{b['oracle_advantage']:.4f})  transfer {b['transfer_accuracy']:.4f} "
                    f"(+{b['transfer_advantage']:.4f})  unseen {b['unseen_key_fraction']:.3f}")
    return out


def kind_records(seeds, split, context, **configuration):
    rows = []
    for seed in seeds:
        ep = episode(seed, split, **configuration)
        owner = ep["probes"]["owner"]
        fills = {}
        for i, o in enumerate(owner):
            fills.setdefault(o, colour_at(ep, i))
        for w in ep["probes"]["hierarchy"]:
            if w["id"] not in fills:
                continue                     # entirely covered by its children
            fill = fills[w["id"]]
            size = (w["rect"][2], w["rect"][3])
            key = {"fill colour": fill, "size (w,h)": size, "fill + size": (fill, size)}[context]
            rows.append((key, w["kind"]))
    return rows


def rung3():
    out = {}
    for mode in ("random", "kind"):
        for context in ("fill colour", "size (w,h)", "fill + size"):
            key = f"colour_mode={mode} | {context}"
            out[key] = lookup_bound(
                kind_records(TRAIN, "train", context, resolution=32, widgets=12, nesting=4,
                             colour_mode=mode),
                kind_records(EVAL, "test", context, resolution=32, widgets=12, nesting=4,
                             colour_mode=mode))
            b = out[key]
            report(key, f"majority {b['majority_baseline']:.4f}  oracle {b['oracle_accuracy']:.4f} "
                        f"(+{b['oracle_advantage']:.4f})  transfer {b['transfer_accuracy']:.4f} "
                        f"(+{b['transfer_advantage']:.4f})")
    return out


def rung4():
    """Is the parent determined by the rectangle set alone?"""
    out = {}
    for name, configuration in (("6 widgets, nesting 2", {}),
                                ("12 widgets, nesting 4", {"resolution": 32, "widgets": 12,
                                                           "nesting": 4}),
                                ("24 widgets, nesting 6", {"resolution": 64, "widgets": 24,
                                                           "nesting": 6})):
        hit = total = ambiguous = 0
        for seed in EVAL:
            ep = episode(seed, "test", **configuration)
            rows = ep["probes"]["hierarchy"]
            for w in rows:
                if w["parent"] == 255:
                    continue
                x, y, ww, hh = w["rect"]
                containers = [o for o in rows if o["id"] != w["id"]
                              and o["rect"][0] <= x and o["rect"][1] <= y
                              and o["rect"][0] + o["rect"][2] >= x + ww
                              and o["rect"][1] + o["rect"][3] >= y + hh]
                if not containers:
                    total += 1
                    continue
                areas = sorted(o["rect"][2] * o["rect"][3] for o in containers)
                ambiguous += len(areas) > 1 and areas[0] == areas[1]
                best = min(containers, key=lambda o: (o["rect"][2] * o["rect"][3], o["id"]))
                total += 1
                hit += best["id"] == w["parent"]
        out[name] = {"widgets": total, "smallest_container_correct": hit / max(1, total),
                     "ties": ambiguous}
        report(f"parent from geometry | {name}",
               f"{hit}/{total} = {hit / max(1, total):.4f}  ties {ambiguous}")
    return out


def main():
    result = {"train_seeds": list(TRAIN), "eval_seeds": list(EVAL)}
    print("\n--- R1: widget edge from a two-pixel neighbourhood ---")
    result["rung1_edge"] = rung1()
    print("\n--- R1b: the same target from the eq-pattern the algebra can express ---")
    result["rung1_edge_pattern"] = rung1(pattern=True)
    print("\n--- R2: glyph from its bounding box ---")
    result["rung2_glyph"] = rung2()
    print("\n--- R3: widget kind from rendered appearance ---")
    result["rung3_kind"] = rung3()
    print("\n--- R4: parent from geometry ---")
    result["rung4_parent"] = rung4()
    dump("bounds", result)


if __name__ == "__main__":
    main()
