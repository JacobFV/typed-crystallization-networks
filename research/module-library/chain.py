"""The three-stage perceptual chain, where each stage's output is the next stage's primitive.

    stage 1  perception.foreground   raw pixels          -> foreground/background
    stage 2  perception.edge         stage 1's module    -> two-position boundary
    stage 3  perception.region       stage 2's module    -> a region-level predicate

Every scaffold here is built from operators already in `tcn.operators`, and the
only thing that differs between an inheriting arm and a non-inheriting one is
*which crystallized modules are available as candidates*. Nothing else about the
data, the supervision, the tolerance or the search changes, so the difference in
search cost is attributable to inheritance and nothing else.

Stages 1 and 2 are the demonstrated capability from `research/discrete-perception`,
reused rather than reinvented: `rung3_mask.module_scaffold` and
`rung35_window.staged_scaffold`/`flat_scaffold` are imported, not copied. Stage 3
is new and is the point of the chain -- an aggregate over a *region* of stage 2's
per-position output, which needs `map`/`filter`/`count` and therefore needs a
module to exist at all.

Hand-initializations, stated as AGENTS.md requires:

* The comparison pool for stage 1 is `rung3_mask.POOL`, five byte values of which
  three are the renderer's background. The full-alphabet variant is the shipped
  `tcn demo --only segmentation` and is not re-run here.
* Stage 2's offset pool is `rung35_window.offsets(R)`, three relative addresses.
* Stage 3's tile geometry (2x2 pixel tiles, last tile row dropped so `pos+1`
  stays inside the image) is declared, not learned; the readout comparison and
  its threshold are searched.
* No arm's answer is supplied: every arm is settled by exhaustive enumeration
  over the declared space, which is a certificate rather than an argument.
"""
from __future__ import annotations

import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "discrete-perception"))

from common import (BYTE, IDX, Builder, bytes_type, episode,  # noqa: E402
                    exact_error, pixel_starts, record_type)
from rung3_mask import POOL, module_scaffold, pixel_examples, signals as fg_signals  # noqa: E402
from rung35_window import (edge_signals, flat_scaffold, offsets, staged_scaffold,  # noqa: E402
                           window_examples)
from tcn.graph import Signal  # noqa: E402
from tcn.types import BOOL, Value, integer, product, setof  # noqa: E402

COUNT = integer(32, signed=False)
COMPARISONS = ("gt", "ge", "lt", "le", "eq")
THRESHOLDS = (0, 1, 2, 3)
TILE = 2
EDGE_STEP = 3                    # one pixel to the right, in bytes


# --------------------------------------------------------------------------
# stage 3: a region-level predicate over stage 2's output
# --------------------------------------------------------------------------

def tile_origins(resolution, tile=TILE):
    """Byte index of every tile's top-left pixel.

    The last tile row is dropped so that the `pos + 1` neighbour every edge
    predicate reads stays inside the image; an address running off the end is a
    property of the candidate, not of the target, and it would silently remove
    conforming programs from the sweep.
    """
    rows = resolution // tile - 1
    return tuple(3 * ((r * tile) * resolution + c * tile)
                 for r in range(rows) for c in range(resolution // tile))


def tile_offsets(resolution, tile=TILE):
    """Byte offsets of the pixels inside one tile, relative to its origin."""
    return tuple(3 * (dr * resolution + dc) for dr in range(tile) for dc in range(tile))


def region_labels(probes, resolution, tile=TILE):
    """`any(fg(i) != fg(i+1) for i in tile)` -- does this tile straddle a boundary?"""
    ids = probes["object_ids"]
    out = {}
    for origin in tile_origins(resolution, tile):
        base = origin // 3
        out[origin] = any((ids[base + d // 3] >= 0) != (ids[base + d // 3 + 1] >= 0)
                          for d in tile_offsets(resolution, tile))
    return out


def region_examples(seeds, resolution, split="train", tile=TILE, seed=0, objects=6):
    REC = record_type(resolution)
    rng = random.Random(seed)
    out = []
    for s in seeds:
        pixels, probes = episode(s, resolution, split=split, objects=objects)
        raw = Value.of(bytes_type(resolution), pixels).raw
        labels = region_labels(probes, resolution, tile)
        for origin, label in labels.items():
            out.append({"inputs": {"rec": Value(REC, (origin, raw))},
                        "targets": {"region": Value.of(BOOL, label)}})
    rng.shuffle(out)
    return out


def region_signals():
    return (Signal("region", "region", ("core",), BOOL, "bce"),)


def keep_flag_module(registry):
    """`(index, flag) -> flag`, the predicate `filter` needs."""
    b = Builder(registry, (("rec", product(IDX, BOOL)),))
    b.add("flag", "project", ["rec"], params={"index": 1})
    return b.program((("y", "flag"),))


def _wrapper_ports(resolution):
    """`(offset, (origin, image))` -- what one mapped record carries."""
    return product(IDX, record_type(resolution))


def wrapper_over_edge(registry, resolution, edge_name):
    """Stage-3 call site for an inherited stage-2 module.

    `(offset, (origin, image)) -> (offset, edge)`. It computes its own absolute
    address from the origin it was handed and calls the frozen edge module
    there. Every node is single-candidate, so the wrapper is itself crystallized
    and registrable; the *choice* stage 3 makes is which wrapper to map.
    """
    b = Builder(registry, (("arg", _wrapper_ports(resolution)),))
    b.add("off", "project", ["arg"], params={"index": 0})
    b.add("rec", "project", ["arg"], params={"index": 1})
    b.add("origin", "project", ["rec"], params={"index": 0})
    b.add("obs", "project", ["rec"], params={"index": 1})
    b.add("pos", "add", ["origin", "off"])
    b.add("here", "tuple", ["pos", "obs"])
    b.add("edge", edge_name, ["here"])
    b.add("flag", "project", ["edge"], params={"index": 1})
    b.add("out", "tuple", ["off", "flag"])
    return b.program((("y", "out"),))


def wrapper_over_foreground(registry, resolution, fg_name, step, truth):
    """The same call site with stage 2 *not* inherited: the edge is re-derived.

    Two calls to the stage-1 foreground module and one Boolean combinator, which
    is exactly stage 2's body. One wrapper per (offset, combinator) pair, so the
    stage-2 structure becomes part of stage 3's search rather than a given.
    """
    b = Builder(registry, (("arg", _wrapper_ports(resolution)),),
                (("step", Value.of(IDX, step)),))
    b.add("off", "project", ["arg"], params={"index": 0})
    b.add("rec", "project", ["arg"], params={"index": 1})
    b.add("origin", "project", ["rec"], params={"index": 0})
    b.add("obs", "project", ["rec"], params={"index": 1})
    b.add("pos", "add", ["origin", "off"])
    b.add("shifted", "add", ["pos", "step"])
    b.add("here_rec", "tuple", ["pos", "obs"])
    b.add("there_rec", "tuple", ["shifted", "obs"])
    b.add("here", fg_name, ["here_rec"])
    b.add("there", fg_name, ["there_rec"])
    b.add("here_flag", "project", ["here"], params={"index": 1})
    b.add("there_flag", "project", ["there"], params={"index": 1})
    b.add("edge", f"truth_{truth}", ["here_flag", "there_flag"])
    b.add("out", "tuple", ["off", "edge"])
    return b.program((("y", "out"),))


def region_scaffold(registry, resolution, wrapper_names, keep_name, tile=TILE):
    """`(origin, image) -> region predicate`, aggregated over the tile.

        held    = insert(empty, rec)                 set[(origin, image)], cap 1
        records = pair(offsets, held)                set[(offset, (origin, image))]
        mapped  = map(records; wrapper)              set[(offset, edge)]
        kept    = filter(mapped; keep_flag)          the edges that fired
        total   = count(kept)
        region  = <comparison>(total, <threshold>)

    Three of those six nodes are the positional-reuse pattern, unchanged. The
    free choices are which wrapper to map (so the inherited module is *selected*,
    not forced) and the readout comparison and its threshold.
    """
    REC = record_type(resolution)
    ARG = _wrapper_ports(resolution)
    offs = tile_offsets(resolution, tile)
    holder = setof(REC, 1)
    locations = setof(IDX, len(offs))
    consts = (("positions", Value.of(locations, offs)), ("empty", Value.of(holder, ())))
    consts += tuple((f"k{k}", Value.of(COUNT, k)) for k in THRESHOLDS)
    b = Builder(registry, (("rec", REC),), consts)
    b.add("held", "insert", ["empty", "rec"])
    b.add("records", "pair", ["positions", "held"])
    b.choice("mapped", [("map", ("records",), {"module": m}) for m in wrapper_names])
    b.add("kept", "filter", ["mapped"], params={"module": keep_name})
    b.add("total", "count", ["kept"])
    b.choice("region", [(c, ("total", f"k{k}"), None) for c in COMPARISONS for k in THRESHOLDS])
    return b.program((("y", "region"),))


# --------------------------------------------------------------------------
# re-exports so the runner reads as one chain
# --------------------------------------------------------------------------

__all__ = ["COUNT", "COMPARISONS", "EDGE_STEP", "POOL", "THRESHOLDS", "TILE", "Builder",
           "bytes_type", "edge_signals", "episode", "exact_error", "fg_signals",
           "flat_scaffold", "keep_flag_module", "module_scaffold",
           "offsets", "pixel_examples", "pixel_starts", "record_type", "region_examples",
           "region_labels", "region_scaffold", "region_signals", "staged_scaffold",
           "tile_offsets", "tile_origins", "window_examples", "wrapper_over_edge",
           "wrapper_over_foreground"]
