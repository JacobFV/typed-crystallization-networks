"""What is legal on an uncommitted byte, before and after the commitment.

Two tables, both produced by asking `Registry.resolve` directly:

1. the operator inventory of ARCHITECTURE section 2, resolved against
   `role="byte"`, against `role="intensity"` and against `role="category"`;
2. every declared conversion *out of* `role="byte"`, with its gradient.

Table 1's `byte` column is unchanged by this branch -- that is the point.  The
uncommitted carrier stays exactly as restricted as it was; what changed is that
there is now a declared exit, and the exit's gradient depends on which
commitment is made.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import BYTE, FMAG, FPLAIN, IDX, MAG8, NOM8, PLAIN8, dump, report
from tcn.operators import BINARY, COMPARE, UNARY, Registry
from tcn.types import BOOL, integer, product

r = Registry()

FAMILIES = {
    "integer/discrete": sorted(BINARY - {"div", "pow", "atan2"}),
    "analytic": ["div", "pow", "atan2"],
    "unary": sorted(UNARY),
    "compare": sorted(COMPARE),
    "aggregation": ["sum", "mean", "reduce_min", "reduce_max"],
    "structure": ["index", "tuple"],
}


def legal(name, types, output=None, params=None):
    try:
        op = r.resolve(name, types, output, params)
        return op.gradient
    except Exception:
        return None


def scalar_row(name, t):
    if name in BINARY:
        return legal(name, (t, t))
    if name in UNARY:
        return legal(name, (t,))
    if name in COMPARE:
        return legal(name, (t, t))
    if name in {"sum", "mean", "reduce_min", "reduce_max"}:
        return legal(name, (product(t, t, t),))
    if name == "index":
        return legal(name, (product(t, t, t), IDX))
    if name == "tuple":
        return legal(name, (t, t))
    raise KeyError(name)


def main():
    result = {}
    columns = [("byte (uncommitted)", BYTE), ("intensity (magnitude)", MAG8),
               ("category (nominal)", NOM8)]
    table = {}
    for family, names in FAMILIES.items():
        for name in names:
            table[name] = {label: scalar_row(name, t) for label, t in columns}
    result["operator_legality"] = table
    print("operator legality by declared role (gradient, or '-' where illegal)")
    print(f"  {'operator':<14s} {'byte':>12s} {'intensity':>12s} {'category':>12s}")
    for family, names in FAMILIES.items():
        for name in names:
            row = table[name]
            print(f"  {name:<14s} " + " ".join(
                f"{(row[label] or '-'):>12s}" for label, _ in columns))
    counts = {label: sum(1 for v in table.values() if v[label]) for label, _ in columns}
    result["legal_counts"] = counts
    result["total_operators"] = len(table)
    print()
    report("operators legal, of " + str(len(table)),
           " / ".join(f"{label}: {counts[label]}" for label, _ in columns))

    # ---- exits from the uncommitted role ----
    exits = {
        "encode  byte -> int[16] byte": legal("encode", (BYTE,), integer(16, signed=False, role="byte")),
        "encode  byte -> int[16] plain": legal("encode", (BYTE,), integer(16, signed=False)),
        "decode  byte -> float32 plain": legal("decode", (BYTE,), FPLAIN),
        "decode  byte -> float32 intensity": legal("decode", (BYTE,), FMAG),
        "quantize byte -> bool": legal("quantize", (BYTE,), BOOL),
        "dequantize byte -> float32 plain": legal("dequantize", (BYTE,), FPLAIN),
        "pack    (byte,) -> int[8] plain": legal("pack", (product(BYTE),), PLAIN8),
        "unpack  int[8] plain -> (byte,)": legal("unpack", (PLAIN8,), product(BYTE)),
        "interpret byte -> int[8] intensity": legal("interpret", (BYTE,), MAG8),
        "interpret byte -> int[8] category": legal("interpret", (BYTE,), NOM8),
        "interpret byte -> int[8] plain": legal("interpret", (BYTE,), PLAIN8),
        "interpret byte -> int[16] intensity": legal("interpret", (BYTE,), integer(16, signed=False, role="intensity")),
        "interpret byte -> float32 intensity": legal("interpret", (BYTE,), FMAG),
        "interpret category -> intensity": legal("interpret", (NOM8,), MAG8),
        "interpret intensity -> category": legal("interpret", (MAG8,), NOM8),
        "interpret byte -> byte": legal("interpret", (BYTE,), BYTE),
        "decode  intensity -> float32 intensity": legal("decode", (MAG8,), FMAG),
    }
    result["conversions"] = exits
    print()
    print("conversions involving the uncommitted role")
    for k, v in exits.items():
        print(f"  {k:<40s} {v or 'ILLEGAL'}")

    # The certificate the demonstration rests on: before this branch the only
    # legal exit was `pack`, and `pack` is a hard gradient boundary.
    differentiable = [k for k, v in exits.items()
                      if k.startswith(("pack", "interpret")) and v not in (None, "none")]
    result["differentiable_exits_from_byte"] = differentiable
    print()
    report("differentiable exits from role=byte", differentiable or "NONE")
    dump("legality", result)


if __name__ == "__main__":
    main()
