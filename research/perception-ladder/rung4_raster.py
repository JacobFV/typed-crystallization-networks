"""Rung 4 -- `raster_text`: pixels to symbol.

Measured facts that shape this rung:
  * the `text` probe has type product(int[32] bounds(0,64), product(BYTE x 64)).
    Every one of its 64 byte fields must be produced by a node of type BYTE, and
    the only BYTE-valued operators are `project` (copy an image byte), `index`
    and constants.  With a 192-byte canvas and a 256-value constant alphabet
    that is 448**64 ~ 1e170 discrete programs and there is no aggregation
    operator with which to condition them jointly.  Full OCR is not merely hard
    here, it is not addressable by the candidate algebra.
  * so the rung is reduced to a two-symbol discrimination whose target is an
    equivalence class of the `text` probe.
  * at an 8x8 canvas, over 60 seeds x 2 symbols: there is NO single byte value
    whose equality test separates 'a' from 'b' (0 candidates), and there ARE
    three bytes separated by a threshold (indices 180-182, threshold 68).
    `eq` is the only differentiable predicate on bytes; the threshold route runs
    through `pack`, which declares gradient="none".
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tcn.generation import Host, Value, BYTE, SCALAR, read_text
from tcn.graph import Program, Node, Candidate, Signal
from tcn.operators import Registry
from tcn.types import product, integer, BOOL

W = H = 8
SYMBOLS = ("a", "b")


def image_type(w=W, h=H):
    return product(integer(16, signed=False), integer(16, signed=False), integer(8, signed=False),
                   product(*(BYTE for _ in range(3 * w * h))))


def examples(seeds, w=W, h=H, symbols=SYMBOLS):
    out = []
    for seed in seeds:
        for i, txt in enumerate(symbols):
            g = Host.create("raster_text", seed=seed, configuration={"text": txt, "width": w, "height": h})
            rec = g.records[-1]
            out.append({"inputs": {"pixels": rec.actor_view().observations["pixels"]},
                        # equivalence class of the `text` probe -- privileged supervision
                        "targets": {"is_b": Value.of(BOOL, read_text(rec.probes["text"]) == symbols[-1]),
                                    "size": rec.latent_states["layout"]}})
    return out


def eq_program(r, w=W, h=H, pool=tuple(range(256)), free_address=True, address=180):
    """The differentiable route: one byte, one equality test against a byte constant."""
    IT = image_type(w, h); BT = IT.items[3]
    addrs = range(3 * w * h) if free_address else (address,)
    nodes = [
        Node("bytes", BT, (Candidate(r.resolve("project", (IT,), BT, {"index": 3}), ("pixels",)),), "obs", 1),
        Node("px", BYTE, tuple(Candidate(r.resolve("project", (BT,), BYTE, {"index": j}), ("bytes",)) for j in addrs), "address", 2),
        Node("hit", BOOL, tuple(Candidate(r.resolve("eq", (BYTE, BYTE)), ("px", f"k{v}")) for v in pool), "compare", 3),
    ]
    return Program((("pixels", IT),), tuple(nodes), (("is_b", "hit"),),
                   tuple((f"k{v}", Value.of(BYTE, v)) for v in pool))


def threshold_program(r, w=W, h=H, free_address=True, address=180, init=128., thr_pool=None):
    """The exact route: pack -> decode -> ordering comparison against a threshold.

    `pack` declares gradient="none", so the address choice receives no gradient at
    all.  With `thr_pool=None` the threshold is a single trainable float constant
    (the differentiable arm); with a pool it becomes an ordinary discrete choice
    (the enumerable arm)."""
    IT = image_type(w, h); BT = IT.items[3]
    U8 = integer(8, signed=False)
    addrs = range(3 * w * h) if free_address else (address,)
    nodes = [
        Node("bytes", BT, (Candidate(r.resolve("project", (IT,), BT, {"index": 3}), ("pixels",)),), "obs", 1),
        Node("px", BYTE, tuple(Candidate(r.resolve("project", (BT,), BYTE, {"index": j}), ("bytes",)) for j in addrs), "address", 2),
        Node("wrap", product(BYTE), (Candidate(r.resolve("tuple", (BYTE,)), ("px",)),), "convert", 3),
        Node("packed", U8, (Candidate(r.resolve("pack", (product(BYTE),), U8), ("wrap",)),), "convert", 4),
        Node("value", SCALAR, (Candidate(r.resolve("decode", (U8,), SCALAR), ("packed",)),), "convert", 5),
    ]
    if thr_pool is None:
        nodes.append(Node("hit", BOOL, (Candidate(r.resolve("lt", (SCALAR, SCALAR)), ("value", "thr")),
                                        Candidate(r.resolve("gt", (SCALAR, SCALAR)), ("value", "thr"))), "compare", 6))
        consts = (("thr", Value.of(SCALAR, init)),); trainable = ("thr",)
    else:
        nodes.append(Node("hit", BOOL, tuple(Candidate(r.resolve(op, (SCALAR, SCALAR)), ("value", f"t{v}"))
                                             for op in ("lt", "gt") for v in thr_pool), "compare", 6))
        consts = tuple((f"t{v}", Value.of(SCALAR, float(v))) for v in thr_pool); trainable = ()
    return Program((("pixels", IT),), tuple(nodes), (("is_b", "hit"),), consts,
                   trainable_constants=trainable)


SIGNAL = (Signal("hit", "is_b", ("compare",), BOOL, "bce"),)
