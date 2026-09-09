"""What the proposed dedup costs when rows do NOT repeat.

A cache only pays if keys recur. The worst case for the proposed diff is a
module over a continuous encoding, where every row is distinct and the dict is
pure overhead. Measure that penalty so the proposal is not one-sided.
"""
import json, statistics, time
from pathlib import Path
import torch
from tcn.types import integer, BOOL
from tcn.operators import Registry
from tcn.graph import Program, Node, Candidate
from tcn.learning import exact_tensor
from proposed_dedup import exact_tensor_dedup, med
import common as C

r = Registry()
I8 = integer(8, signed=False)
# a module over int[8]: 256 distinct values per input, so a 512-row batch of
# random arguments repeats far less than a Boolean one
body = Program((("a", I8), ("b", I8)),
               (Node("g", I8, (Candidate(r.resolve("add", (I8, I8)), ("a", "b")),), "core", 1, 0),),
               (("out", "g"),)).validate(r)
name = r.register_module(body)
op = r.resolve(name, (I8, I8))
out = {}
for batch, label in ((64, "64 rows"), (512, "512 rows")):
    xs = [torch.randint(0, 128, (batch, 1)).float() for _ in range(2)]
    distinct = len({(float(xs[0][i, 0]), float(xs[1][i, 0])) for i in range(batch)})
    assert torch.equal(exact_tensor(r, op, xs), exact_tensor_dedup(r, op, xs))
    a = med(lambda: exact_tensor(r, op, xs), reps=20)
    b = med(lambda: exact_tensor_dedup(r, op, xs), reps=20)
    out[label] = {"batch": batch, "distinct_rows": distinct,
                  "shipped_us": round(a, 1), "deduped_us": round(b, 1),
                  "ratio": round(b / a, 3)}
# and the Boolean case for contrast, on the experiment's own tensors
rb = Registry(); nb = rb.register_module(C.minimal_module(rb, "maj"))
ob = rb.resolve(nb, (BOOL, BOOL, BOOL))
import itertools
rows = list(itertools.product((0., 1.), repeat=6))
cols = {k: torch.tensor([[row[i]] for row in rows]) for i, k in enumerate(C.INPUTS)}
xs = [cols[k] for k in ("a", "b", "c")]
a = med(lambda: exact_tensor(rb, ob, xs), reps=20)
b = med(lambda: exact_tensor_dedup(rb, ob, xs), reps=20)
out["boolean module, 64 rows"] = {"batch": 64, "distinct_rows": 8,
                                  "shipped_us": round(a, 1), "deduped_us": round(b, 1),
                                  "ratio": round(b / a, 3)}
Path("dedup_overhead.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
