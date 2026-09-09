"""What the F2 memoization actually bought, measured against its own baseline.

`costs.py` shows the module candidate is still 13-184x a primitive candidate in
the soft graph. The handoff claim was "4.0x (was 27.4x)". Both can be true only
if they measure different things, so measure every version explicitly:

  D1  one module call at batch one vs one primitive at batch one -- the "5-node
      module call is 4.0x a primitive" comparison.
  D2  what the memoization removed: the cost of `Program.validate` per row,
      measured directly, and the reconstructed pre-fix per-row cost.
  D3  the residual: how much of a module call is the O(batch) Python loop that
      remains, i.e. the ceiling the current fix leaves in place.
  D4  row deduplication -- a Boolean module has at most 2**k distinct argument
      rows, so a 64-row batch of a 3-input module contains 8 distinct calls.
"""
from __future__ import annotations

import itertools
import json
import statistics
import time
from pathlib import Path

import torch

from tcn.types import BOOL, Value
from tcn.operators import Registry
from tcn.learning import exact_tensor, relaxed
from costs import maj3_body

HERE = Path(__file__).parent


def med(f, reps=200):
    f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e6


def main():
    r = Registry()
    body = maj3_body(r)
    name = r.register_module(body)
    mop = r.resolve(name, (BOOL, BOOL, BOOL))
    xop = r.resolve("xor", (BOOL, BOOL))

    x1 = [torch.rand(1, 1) for _ in range(3)]
    x64 = [torch.rand(64, 1) for _ in range(3)]

    d1 = {
        "module_call_batch1_us": round(med(lambda: exact_tensor(r, mop, x1)), 2),
        "primitive_relaxed_batch1_us": round(med(lambda: relaxed(r, xop, x1[:2])), 2),
        "primitive_exact_tensor_batch1_us": round(med(lambda: exact_tensor(r, xop, x1[:2])), 2),
    }
    d1["ratio_vs_relaxed_primitive"] = round(d1["module_call_batch1_us"] / d1["primitive_relaxed_batch1_us"], 1)
    d1["ratio_vs_exact_primitive"] = round(d1["module_call_batch1_us"] / d1["primitive_exact_tensor_batch1_us"], 1)

    # D2: the memoized validation, measured on its own.
    validate_us = med(lambda: body.validate(r), reps=200)
    args = [Value.of(BOOL, False)] * 3
    run_us = med(lambda: body.run({"a": args[0], "b": args[1], "c": args[2]}, registry=r), reps=200)
    d2 = {
        "validate_us": round(validate_us, 2),
        "memoized_run_us": round(run_us, 2),
        "reconstructed_prefix_run_us": round(run_us + validate_us, 2),
        "speedup_from_memoization": round((run_us + validate_us) / run_us, 2),
    }
    # confirm the memoization is live: a fresh registry forces one re-validation
    fresh = Registry(); fresh.register_module(maj3_body(fresh))
    d2["validation_is_memoized"] = getattr(body, "_validated", None) is r

    # D3: where the remaining time goes at batch 64.
    m64 = med(lambda: exact_tensor(r, mop, x64), reps=50)
    d3 = {
        "module_batch64_us": round(m64, 2),
        "per_row_us": round(m64 / 64, 2),
        "pure_body_run_us_x64": round(run_us * 64, 2),
        "fraction_in_body_execution": round(run_us * 64 / m64, 3),
    }

    # D4: distinct argument rows in a 6-input Boolean batch.
    rows = list(itertools.product((0., 1.), repeat=6))
    xs = [torch.tensor([[row[i]] for row in rows]) for i in range(3)]
    distinct = len({tuple(row[:3]) for row in rows})
    d4 = {
        "batch_rows": len(rows), "distinct_argument_rows": distinct,
        "measured_us": round(med(lambda: exact_tensor(r, mop, xs), reps=50), 2),
        "potential_speedup_from_dedup": round(len(rows) / distinct, 1),
    }

    out = {"D1_batch_one": d1, "D2_memoization": d2, "D3_residual": d3, "D4_dedup": d4}
    (HERE / "costs_detail.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
