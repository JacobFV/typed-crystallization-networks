"""What the operator registry admits on each candidate observation encoding.

Nothing here is an opinion: every cell is `Registry.resolve` accepting or
rejecting a signature, written to `out/operator_legality.json`.

    .venv/bin/python research/external-environments/operator_legality.py
"""
from __future__ import annotations
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tcn.operators import Registry
from tcn.types import BOOL, Type, floating, fixed, integer, product

OUT = pathlib.Path(__file__).parent / "out"

CANDIDATES = {
    "float32 plain (as shipped: orientation)": floating(32, frame="joint"),
    "float32 unit=rad (the mistake)": floating(32, unit="rad", frame="joint"),
    "float32 unit=rad/s (as shipped: rates)": floating(32, unit="rad/s", frame="joint"),
    "float32 bounds=(-1,1)": floating(32, frame="joint", bounds=(-1.0, 1.0)),
    "fixed int[16] scale 4096": fixed(16, 4096, frame="joint"),
    "int[16] plain integer": integer(16, frame="joint"),
    "int[8] role=byte (pixels)": integer(8, signed=False, role="byte"),
}

UNARY = ["neg", "abs", "exp", "log", "sin", "cos", "sqrt"]
BINARY = ["add", "sub", "mul", "div", "pow", "atan2", "min", "max", "mod", "idiv"]
COMPARE = ["eq", "lt", "le", "gt", "ge"]
REDUCE = ["sum", "mean", "reduce_max", "count"]


def probe(registry, name, types, output=None, parameters=None):
    try:
        op = registry.resolve(name, types, output, parameters)
    except Exception as error:  # legality is exactly "resolve did not raise"
        return {"legal": False, "reason": f"{type(error).__name__}: {error}"}
    return {"legal": True, "output": op.output.to_dict(), "gradient": op.gradient}


def main():
    registry = Registry()
    table = {}
    for label, t in CANDIDATES.items():
        row = {}
        for name in UNARY:
            row[name] = probe(registry, name, (t,))
        for name in BINARY:
            row[name] = probe(registry, name, (t, t))
        for name in COMPARE:
            row[name] = probe(registry, name, (t, t))
        for name in REDUCE:
            row[name] = probe(registry, name, (product(t, t, t),))
        # `pack` is the documented escape from a non-numeric carrier.
        row["pack"] = probe(registry, "pack", (product(t,),),
                            output=Type("int", t.bits if t.kind == "int" else 1,
                                        encoding=t.encoding, role=t.role, unit=t.unit, frame=t.frame))
        table[label] = {"type": t.to_dict(), "numeric": t.numeric, "operators": row}

    # Can a united channel ever reach the dimensionless algebra?
    rate = CANDIDATES["float32 unit=rad/s (as shipped: rates)"]
    plain = CANDIDATES["float32 plain (as shipped: orientation)"]
    escapes = {
        "div(rate, rate) -> plain?": probe(registry, "div", (rate, rate)),
        "div output equals the plain angle type": registry.resolve("div", (rate, rate)).output == plain,
        "encode(rate) -> plain (unit change)": probe(registry, "encode", (rate,), output=plain),
        "quantize(rate) -> fixed with same unit": probe(
            registry, "quantize", (rate,), output=fixed(16, 4096, unit="rad/s", frame="joint")),
        "add(rate, plain)": probe(registry, "add", (rate, plain)),
        "mul(rate, rate) unit": registry.resolve("mul", (rate, rate)).output.unit,
    }

    # Refinement bounds pass `resolve` and fail at execution: a total operator
    # becomes partial, which is a runtime error rather than an illegal candidate.
    from tcn.types import Value
    bounded = CANDIDATES["float32 bounds=(-1,1)"]
    execution = {}
    op = registry.resolve("add", (bounded, bounded))
    try:
        registry.exact(op, (Value.of(bounded, 0.8), Value.of(bounded, 0.7)))
        execution["add(0.8, 0.7) on bounds=(-1,1)"] = "returned a value"
    except Exception as error:
        execution["add(0.8, 0.7) on bounds=(-1,1)"] = f"{type(error).__name__}: {error}"
    unbounded = CANDIDATES["float32 plain (as shipped: orientation)"]
    op = registry.resolve("add", (unbounded, unbounded))
    execution["add(0.8, 0.7) unbounded"] = registry.exact(
        op, (Value.of(unbounded, 0.8), Value.of(unbounded, 0.7))).decoded
    op = registry.resolve("mul", (bounded, bounded))
    execution["mul output keeps bounds"] = op.output.bounds
    escapes.update(execution)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "operator_legality.json").write_text(json.dumps({"table": table, "escapes": escapes}, indent=1, sort_keys=True))

    names = UNARY + BINARY + COMPARE + REDUCE + ["pack"]
    width = max(len(x) for x in CANDIDATES) + 2
    print("".ljust(width) + " ".join(n[:6].rjust(6) for n in names))
    for label, row in table.items():
        cells = " ".join(("y" if row["operators"][n]["legal"] else ".").rjust(6) for n in names)
        print(label.ljust(width) + cells)
    print()
    for k, v in escapes.items():
        print(f"  {k}: {v if not isinstance(v, dict) else ('legal' if v['legal'] else v['reason'])}")


if __name__ == "__main__":
    main()
