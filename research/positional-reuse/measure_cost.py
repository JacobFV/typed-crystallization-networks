"""Description size, execution cost and wall time: shared map vs per-position.

Three programs computing the same thing, so the comparison is like for like:

  shared        hold / pair / map, 3 caller nodes, one module definition
  per_position  one module call site per position, 2N caller nodes
  inlined       the module body written out at every position, (k+2)N nodes

`inlined` produces a plain tuple of window values and `shared` produces a set of
(position, value) records, so the two are compared on cost, not on identical
output types; `per_position` calls the very same module `shared` maps, which is
the comparison that decides whether sharing pays.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from shared import (F, IDX, Builder, inlined_program, per_position_program,
                    shared_map_program, wide, window_module)
from tcn.operators import Registry
from tcn.types import Value


def structural_symbols(program, registry, seen=None):
    """Nodes plus constants, counting each module definition once.

    `description_bits` measures serialized JSON length, which for a wide input is
    dominated by the observation type repeated in every node's signature. This
    counts program structure instead, which is what sharing is supposed to save.
    """
    seen = set() if seen is None else seen
    total = len(program.nodes) + len(program.constants)
    for node in program.nodes:
        for c in node.candidates:
            for name in (c.operator.name, dict(c.operator.parameters).get("module", "")):
                if name.startswith("module:") and name not in seen:
                    seen.add(name)
                    total += structural_symbols(registry.modules[name], registry, seen)
    return total


def mean_body(b, reads, suffix=""):
    b.add("win" + suffix, "tuple", reads)
    return b.add("val" + suffix, "mean", ["win" + suffix])


def timed(fn, repeats):
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def run(width, offsets, repeats=3):
    positions = tuple(range(width - max(offsets)))
    n = len(positions)
    r = Registry()
    mod = window_module(r, width, offsets, mean_body)
    name = r.register_module(mod)
    shared = shared_map_program(r, width, positions, [name])
    per = per_position_program(r, width, positions, name)
    inl = inlined_program(r, width, positions, offsets, mean_body)
    W = wide(width)
    x = {"observation": Value.of(W, tuple(float(i) for i in range(width)))}

    rows = {}
    for label, prog in (("shared", shared), ("per_position", per), ("inlined", inl)):
        prog.run(x, registry=r)  # warm the memoised validation
        rows[label] = {
            "caller_nodes": len(prog.nodes),
            "description_bits": prog.description_bits(r),
            "structural_symbols": structural_symbols(prog, r),
            "execution_cost": prog.execution_cost(r),
            "seconds": timed(lambda p=prog: p.run(x, registry=r), repeats),
        }
    rows["module_nodes"] = len(mod.nodes)
    rows["positions"] = n
    rows["width"] = width
    return rows


if __name__ == "__main__":
    out = []
    for width in (6, 10, 18, 34, 66, 130):
        row = run(width, (0, 1, 2))
        out.append(row)
        s, p, i = row["shared"], row["per_position"], row["inlined"]
        print(f"width={row['width']:4d} N={row['positions']:4d} | "
              f"nodes {s['caller_nodes']:3d}/{p['caller_nodes']:4d}/{i['caller_nodes']:4d} | "
              f"symbols {s['structural_symbols']:4d}/{p['structural_symbols']:5d}/{i['structural_symbols']:5d} | "
              f"bits {s['description_bits']:7d}/{p['description_bits']:8d}/{i['description_bits']:8d} | "
              f"cost {s['execution_cost']:8.0f}/{p['execution_cost']:8.0f}/{i['execution_cost']:8.0f} | "
              f"ms {1e3*s['seconds']:8.2f}/{1e3*p['seconds']:8.2f}/{1e3*i['seconds']:8.2f}")
    path = pathlib.Path(__file__).parent / "cost.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)
