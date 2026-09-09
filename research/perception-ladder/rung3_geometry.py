"""Rung 3 -- `geometry`: per-pixel foreground/background from raw RGB bytes.

What the operator algebra permits on an image (measured, see RESULTS.md):
  * `image_value` types every channel as BYTE = int[8] with role="byte", and
    `Type.numeric` excludes role "byte".  So `sum`, `mean`, `min`, `max`, `fft`,
    every arithmetic operator and every ordering comparison are ILLEGAL on
    pixels.  The only operators that consume a pixel are `project`, `index`,
    `eq`, `pack`/`unpack` and `tuple`.
  * `pack` is the only route from bytes to a numeric type, and it declares
    gradient="none": it is a hard boundary, so *nothing behind it can be learned
    by gradient*.
  * `eq` (gradient="surrogate") is therefore the ONLY differentiable predicate
    available on an image.
  * a BYTE constant cannot be a trainable constant (`Program.validate` requires
    `type.numeric`), so a colour value must be searched discretely.

The task: predict whether a pixel is background.  Background is exactly
depth == 0 (`render()` sets depth to 0 where nothing was drawn), so the target
is an equivalence class of the generator's own `depth` probe -- privileged
supervision, never an input.  The reference program is
  hit = eq(red, 24) and eq(green, 30)
which was measured exact on 200 episodes at every resolution tested.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tcn.generation import Host, Action, Value, SCALAR, BYTE
from tcn.graph import Program, Node, Candidate, Signal
from tcn.operators import Registry
from tcn.types import product, integer, BOOL

VEC3 = product(SCALAR, SCALAR, SCALAR)
# The shipped camera puts every object at ~0.07*R pixels across, leaving 98% of
# every image background at every resolution up to 32.  One `camera` action --
# part of the generator's declared action schema, not a code change -- moves the
# eye in and gives a 71/29 background/foreground split.
APPROACH = Action("camera", arguments=(("delta", Value.of(VEC3, (-2.4, -2.2, -3.6))),))
BG = (24, 30, 43)
AND, OR = 8, 14


def image_type(R):
    return product(integer(16, signed=False), integer(16, signed=False), integer(8, signed=False),
                   product(*(BYTE for _ in range(3 * R * R))))


def examples(seeds, R, objects=3, approach=True, dense=False):
    out = []
    for seed in seeds:
        h = Host.create("geometry", seed=seed, configuration={"resolution": R, "objects": objects})
        if approach: h.step([APPROACH])
        rec = h.records[-1]
        pixels = rec.actor_view().observations["pixels"]
        depth = rec.probes["depth"].decoded
        bg = [d == 0. for d in depth]                       # equivalence class of the depth probe
        p = (R * R) // 2
        ex = {"inputs": {"pixels": pixels},
              "targets": {"centre": Value.of(BOOL, bool(bg[p])),
                          "mask": Value.of(product(*(BOOL for _ in bg)), tuple(bg)),
                          "allbg": Value.of(BOOL, all(bg))}}
        if dense:
            for i, b in enumerate(bg): ex["targets"][f"hit{i}"] = Value.of(BOOL, bool(b))
        out.append(ex)
    return out


def _pool_consts(pool):
    return tuple((f"k{v}", Value.of(BYTE, v)) for v in pool)


def _pixel_nodes(r, R, IT, BT, index, name, free_address, pool, and_menu=(AND,)):
    """Two channel probes for output pixel `index`, then their conjunction."""
    nodes = []
    for c, off in ((0, 0), (1, 1)):
        addrs = range(3 * R * R) if free_address else (3 * index + off,)
        nodes.append(Node(f"{name}_px{c}", BYTE,
                          tuple(Candidate(r.resolve("project", (BT,), BYTE, {"index": j}), ("bytes",)) for j in addrs),
                          "address", 2))
        nodes.append(Node(f"{name}_eq{c}", BOOL,
                          tuple(Candidate(r.resolve("eq", (BYTE, BYTE)), (f"{name}_px{c}", f"k{v}")) for v in pool),
                          "compare", 3))
    nodes.append(Node(name, BOOL, tuple(Candidate(r.resolve(f"truth_{t}", (BOOL, BOOL)),
                                                  (f"{name}_eq0", f"{name}_eq1")) for t in and_menu),
                      "logic", 4))
    return nodes


def centre_program(r, R, free_address=True, pool=(BG[0], BG[1]), and_menu=(AND,)):
    """Single output: is the centre pixel background?  Width axis = 3*R*R addresses."""
    IT = image_type(R); BT = IT.items[3]
    nodes = [Node("bytes", BT, (Candidate(r.resolve("project", (IT,), BT, {"index": 3}), ("pixels",)),), "obs", 1)]
    nodes += _pixel_nodes(r, R, IT, BT, (R * R) // 2, "hit", free_address, pool, and_menu)
    return Program((("pixels", IT),), tuple(nodes), (("centre", "hit"),), _pool_consts(pool))


def mask_program(r, R, free_address=False, pool=(BG[0], BG[1]), and_menu=(AND,), head="mask"):
    IT = image_type(R); BT = IT.items[3]
    nodes = [Node("bytes", BT, (Candidate(r.resolve("project", (IT,), BT, {"index": 3}), ("pixels",)),), "obs", 1)]
    for i in range(R * R):
        nodes += _pixel_nodes(r, R, IT, BT, i, f"hit{i}", free_address, pool, and_menu)
    if head == "mask":
        out = product(*(BOOL for _ in range(R * R)))
        nodes.append(Node("mask", out, (Candidate(r.resolve("tuple", tuple(BOOL for _ in range(R * R))),
                                                  tuple(f"hit{i}" for i in range(R * R))),), "readout", 5))
        outputs = (("mask", "mask"),)
    else:                                            # entangled head: AND over every pixel
        prev = "hit0"
        for i in range(1, R * R):
            nodes.append(Node(f"acc{i}", BOOL, tuple(Candidate(r.resolve(f"truth_{t}", (BOOL, BOOL)), (prev, f"hit{i}"))
                                                     for t in (AND, OR)), "readout", 4 + i))
            prev = f"acc{i}"
        outputs = (("allbg", prev),)
    return Program((("pixels", IT),), tuple(nodes), outputs, _pool_consts(pool))


def centre_signal():
    return (Signal("hit", "centre", ("logic",), BOOL, "bce"),)


def mask_signals(R, dense=False):
    out = [Signal("mask", "mask", ("readout",), product(*(BOOL for _ in range(R * R))), "bce")]
    if dense: out += [Signal(f"hit{i}", f"hit{i}", ("logic",), BOOL, "bce") for i in range(R * R)]
    return tuple(out)


def allbg_signals(R, dense=False):
    last = "hit0" if R * R == 1 else f"acc{R*R-1}"
    out = [Signal(last, "allbg", ("readout", "logic"), BOOL, "bce")]
    if dense: out += [Signal(f"hit{i}", f"hit{i}", ("logic",), BOOL, "bce") for i in range(R * R)]
    return tuple(out)
