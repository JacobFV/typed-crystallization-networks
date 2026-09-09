"""How far past the shipped observation widths the pattern still runs.

`geometry` ships `pixels` at 3075 wide with 1024 pixels; `embodied_world` ships
`focus_pixels` at 18435. This walks the exact apply and the module search out to
and beyond both, and stops at the first configuration that exceeds the budget.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch

from measure_scaling import BACKGROUND, bytes_type, scaffold, synthetic
from shared import IDX, shared_map_program
from tcn.graph import Signal
from tcn.learning import SoftProgram, tensor
from tcn.operators import Registry
from tcn.search import enumerate_fit
from tcn.types import BOOL, Value, product

BUDGET = 900.


def run(res, examples=48):
    positions = tuple(3 * i for i in range(res * res))
    width = 3 * res * res
    B = bytes_type(width)
    REC = product(IDX, B)
    r = Registry()
    data = synthetic(width, positions, None)
    obs = Value.of(B, data)
    row = {"resolution": res, "width": width, "positions": len(positions)}

    sc = scaffold(r, width, searched=True)
    labels = [(data[p], data[p + 1], data[p + 2]) != BACKGROUND for p in positions]
    exs = [{"inputs": {"rec": Value.of(REC, (positions[i], data))},
            "targets": {"foreground": Value.of(BOOL, labels[i])}}
           for i in range(min(examples, len(positions)))]
    signals = (Signal("foreground", "foreground", ("core",), BOOL, "bce"),)
    start = time.perf_counter()
    search = enumerate_fit(sc, exs, signals, r, tolerance=1e-6, stop_at_first=True)
    row["enumerate_seconds"] = time.perf_counter() - start
    row["enumerate_solved"] = search.solved

    model = SoftProgram(sc, r)
    inputs = {"rec": torch.stack([tensor(ex["inputs"]["rec"]) for ex in exs])}
    start = time.perf_counter()
    model(inputs)
    row["soft_forward_seconds"] = time.perf_counter() - start

    frozen = scaffold(r, width, searched=False)
    caller = shared_map_program(r, width, positions, [r.register_module(frozen)], element=bytes_type(1).items[0])
    row["caller_nodes"] = len(caller.nodes)
    start = time.perf_counter()
    out, _ = caller.run({"observation": obs}, registry=r)
    row["exact_apply_seconds"] = time.perf_counter() - start
    row["exact_apply_correct"] = out["y"].decoded == frozenset(
        (positions[i], labels[i]) for i in range(len(positions)))
    return row


if __name__ == "__main__":
    rows = []
    for res in (32, 48, 64, 79, 96, 128):
        row = run(res)
        rows.append(row)
        print(f"res={row['resolution']:4d} W={row['width']:6d} N={row['positions']:6d} | "
              f"enum {row['enumerate_seconds']:8.2f}s solved={row['enumerate_solved']} | "
              f"soft-fwd {row['soft_forward_seconds']:7.2f}s | "
              f"apply {row['exact_apply_seconds']:9.2f}s ok={row['exact_apply_correct']}")
        sys.stdout.flush()
        if row["exact_apply_seconds"] > BUDGET:
            print(f"stopping: exact apply exceeded the {BUDGET:.0f}s budget")
            break
    path = pathlib.Path(__file__).parent / "wide.json"
    path.write_text(json.dumps(rows, indent=1))
    print("wrote", path)
