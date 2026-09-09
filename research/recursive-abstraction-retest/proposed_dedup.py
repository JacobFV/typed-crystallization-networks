"""Measure the proposed `exact_tensor` row-deduplication before proposing it.

F2's fix memoized `Program.validate` per (program, registry), which removed the
33.95 us of revalidation that dominated a module call. What remains is the
O(batch) Python loop: `exact_tensor` calls `Registry.exact` once per row, and at
batch 64 the module candidate is still ~105x a primitive candidate (`costs.py`).

But exact execution is a pure function of the row, and a typed row over finite
carriers repeats: a 3-input Boolean module has at most 8 distinct argument rows
no matter how large the batch. A 64-row truth table therefore contains 8 real
calls and 56 repeats.

This measures the deduplicated version against the shipped one on exactly the
tensors the experiment feeds it, so the proposed diff in RESULTS.md carries a
measured number rather than an argument. The core file is NOT modified: the
variant is a local copy.
"""
from __future__ import annotations

import itertools
import json
import math
import statistics
import time
from pathlib import Path

import torch

from tcn.types import BOOL, Value
from tcn.operators import Registry
from tcn.learning import exact_tensor

import common as C

HERE = Path(__file__).parent


def exact_tensor_dedup(registry, op, xs):
    """`tcn.learning.exact_tensor` with the pure-function repeat removed."""
    shape = torch.broadcast_shapes(*(x.shape[:-1] for x in xs)) if xs else ()
    batch = math.prod(shape) if shape else 1
    flat = [x.detach().expand(*shape, x.shape[-1]).reshape(batch, -1).cpu().tolist() for x in xs]
    cache = {}
    vals = []
    for i in range(batch):
        key = tuple(tuple(x[i]) for x in flat)
        got = cache.get(key)
        if got is None:
            args = [Value.unflat(t, x[i]) for t, x in zip(op.inputs, flat)]
            got = cache[key] = registry.exact(op, args).flat()
        vals.append(got)
    device = xs[0].device if xs else None
    return torch.tensor(vals, dtype=torch.float32, device=device).reshape(*shape, op.output.width)


def med(f, reps=20):
    f()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e6


def main():
    r = Registry()
    name = r.register_module(C.minimal_module(r, "maj"))
    op = r.resolve(name, (BOOL, BOOL, BOOL))

    rows = list(itertools.product((0., 1.), repeat=6))
    out = {"per_binding": [], "equivalence_verified": True}
    for binding in (("a", "b", "c"), ("a", "a", "b"), ("a", "b", "a")):
        cols = {k: torch.tensor([[row[i]] for row in rows]) for i, k in enumerate(C.INPUTS)}
        xs = [cols[k] for k in binding]
        a = exact_tensor(r, op, xs)
        b = exact_tensor_dedup(r, op, xs)
        out["equivalence_verified"] &= bool(torch.equal(a, b))
        shipped = med(lambda: exact_tensor(r, op, xs))
        deduped = med(lambda: exact_tensor_dedup(r, op, xs))
        out["per_binding"].append({
            "binding": list(binding), "batch": len(rows),
            "distinct_rows": len({tuple(float(x[i, 0]) for x in xs) for i in range(len(rows))}),
            "shipped_us": round(shipped, 1), "deduped_us": round(deduped, 1),
            "speedup": round(shipped / deduped, 2)})

    # what it would do to a whole arm-B forward pass: every module candidate at
    # every node, over the six inputs, on the 64-row truth table
    cols = {k: torch.tensor([[row[i]] for row in rows]) for i, k in enumerate(C.INPUTS)}
    bindings = list(itertools.product(C.INPUTS, repeat=3))
    nodes = 9

    def sweep(fn):
        for b in bindings:
            fn(r, op, [cols[k] for k in b])

    shipped = med(lambda: sweep(exact_tensor), reps=5) / 1e6
    deduped = med(lambda: sweep(exact_tensor_dedup), reps=5) / 1e6
    out["arm_B_forward_pass"] = {
        "module_candidates_per_node": len(bindings), "nodes": nodes,
        "shipped_seconds_per_step": round(shipped * nodes, 3),
        "deduped_seconds_per_step": round(deduped * nodes, 3),
        "speedup": round(shipped / deduped, 2),
        "measured_arm_B_seconds_per_step": 3.29,
        "measured_arm_A_seconds_per_step": 0.17,
    }
    (HERE / "proposed_dedup.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
