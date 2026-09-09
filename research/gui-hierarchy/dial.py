"""Is the difficulty dial real?

`generators/logic`'s `depth` looked like a difficulty axis and was not: it varied
candidate-set size while target complexity stayed flat, and 40% of its depth-8
targets were constant functions (FINDINGS F-bench).  That invalidated a
97%-at-every-depth result.  So every dial this generator declares is measured
here on four axes rather than asserted:

  structure    achieved widget count and tree depth against the request
  appearance   distinct colours actually on the screen, and the majority
               baseline of the rung-1 target
  ceiling      the two-pixel recoverability oracle from `bounds.py`
  search       the rung-1 space size and how many of its programs conform

The last two together are the point.  A dial that moves the ceiling below 1.0
makes the rung impossible and the search must return zero conforming programs; a
dial that leaves the ceiling at 1.0 but moves the conforming count is a search
difficulty dial; a dial that moves neither is not a difficulty dial at all and is
reported as such.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from bounds import edge_records
from common import (all_conforming, colour_at, dump, edge_positions, episode, lookup_bound,
                    report)
from rung1_edges import examples, narrow_offsets, scaffold, signals
from tcn.operators import Registry
from tcn.search import space_size

TRAIN = tuple(range(8))
EVAL = tuple(range(100, 108))

DIALS = {
    "baseline: 6 widgets, nesting 2, palette 32, R=16": {},
    "widgets 2": {"widgets": 2},
    "widgets 12": {"widgets": 12},
    "widgets 20, nesting 5, R=32": {"widgets": 20, "nesting": 5, "resolution": 32},
    "nesting 1": {"widgets": 12, "nesting": 1},
    "nesting 4": {"widgets": 12, "nesting": 4},
    "palette 4": {"palette": 4},
    "palette 4, widgets 12": {"palette": 4, "widgets": 12},
    "palette 64": {"palette": 64},
    "resolution 32": {"resolution": 32},
    "resolution 48": {"resolution": 48},
    "borders": {"borders": True},
    "labels, R=32": {"resolution": 32, "labels": True},
    "colour_mode=kind": {"colour_mode": "kind"},
    "rung-1 screen: 12 widgets, nesting 4, min_size 4": {"widgets": 12, "nesting": 4,
                                                         "min_size": 4},
    "min_size 3, widgets 12, nesting 4": {"widgets": 12, "nesting": 4, "min_size": 3},
    "rung-1 screen, palette_levels 16": {"widgets": 12, "nesting": 4, "min_size": 4,
                                         "palette_levels": 16},
    "rung-1 screen, palette_levels 32": {"widgets": 12, "nesting": 4, "min_size": 4,
                                         "palette_levels": 32},
}


def structure(seeds, **configuration):
    counts, depths, colours = [], [], []
    for seed in seeds:
        ep = episode(seed, "train", **configuration)
        rows = {w["id"]: w for w in ep["probes"]["hierarchy"]}
        d = {}
        for i in sorted(rows):
            d[i] = 0 if rows[i]["parent"] == 255 else d[rows[i]["parent"]] + 1
        counts.append(len(rows))
        depths.append(max(d.values()))
        colours.append(len({colour_at(ep, i) for i in range(ep["width"] * ep["height"])}))
    return {"widgets_mean": sum(counts) / len(counts), "widgets_max": max(counts),
            "tree_depth_max": max(depths), "distinct_colours_mean": sum(colours) / len(colours),
            "observation_bytes": 3 * ep["width"] * ep["height"],
            "positions_per_screen": len(edge_positions(ep))}


def main():
    registry = Registry()
    result = {}
    for name, configuration in DIALS.items():
        row = structure(TRAIN, **configuration)
        row["configuration"] = configuration
        bound = lookup_bound(edge_records(TRAIN, "train", "x", **configuration),
                             edge_records(EVAL, "test", "x", **configuration))
        row["majority_baseline"] = bound["majority_baseline"]
        row["oracle_ceiling"] = bound["oracle_accuracy"]
        probe = episode(0, "train", **configuration)
        program = scaffold(registry, probe, narrow_offsets(probe["width"]), False)
        train = examples(TRAIN, "train", 24, seed=7, **configuration)
        survey = all_conforming(program, train, signals(), registry, 1e-6)
        survey.pop("conforming")
        row["rung1_space"] = space_size(program)
        row["rung1_conforming"] = survey["count"]
        row["rung1_density"] = survey["count"] / survey["space_size"]
        row["rung1_seconds"] = survey["seconds"]
        result[name] = row
        report(name, f"widgets {row['widgets_mean']:.1f} depth {row['tree_depth_max']} "
                     f"colours {row['distinct_colours_mean']:.1f} bytes {row['observation_bytes']} "
                     f"| majority {row['majority_baseline']:.4f} ceiling {row['oracle_ceiling']:.4f} "
                     f"| space {row['rung1_space']} conforming {row['rung1_conforming']}")
    dump("dial", result)


if __name__ == "__main__":
    main()
