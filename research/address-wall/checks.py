"""The measurement warnings, verified here, plus one defect they led to.

1. `SoftProgram` zero-initialises every choice logit, so `torch.manual_seed`
   does not vary synthesis.
2. `relaxed("tuple", ...)` is a bare `torch.cat` that will not broadcast a
   batched value against an unbatched constant.
3. `eq`'s surrogate underflows to exactly 0.0 in float32 at |a-b| >= 11.
4. NEW: `relaxed("index", ...)` at the shipped temperature of 1.0 does not agree
   with `Registry.exact` even at an exactly integer address.  The softmax kernel
   puts only 0.564 of its mass on the addressed element, so the relaxation does
   not reduce to the operator it relaxes at any point of its own domain.
5. NEW: `SoftProgram` keeps one temperature per node and divides BOTH the
   candidate softmax AND the operator's relaxation by it, so the two cannot be
   tuned independently at a node that has a real choice.

Run:  .venv/bin/python research/address-wall/checks.py
"""
from __future__ import annotations

import math

import torch

from tcn.graph import Candidate, Node, Program, Signal
from tcn.learning import SoftProgram, relaxed, tensor
from tcn.operators import Registry
from tcn.types import BOOL, Value, floating, integer, product

from instrument import ADDR, V, dump


def warning_1_zero_init():
    r = Registry()
    arr_t = product(*([V] * 8))
    cands = tuple(Candidate(r.resolve("project", (arr_t,), V, {"index": i}), ("arr",))
                  for i in range(8))
    p = Program((("arr", arr_t),), (Node("a", V, cands),), (("y", "a"),)).validate(r)
    inits = []
    for seed in range(4):
        torch.manual_seed(seed)
        inits.append(SoftProgram(p, r).choices[0].detach().tolist())
    return {"four_seeds_identical": all(x == inits[0] for x in inits),
            "all_zero": all(v == 0. for v in inits[0]),
            "init": inits[0]}


def warning_2_tuple_broadcast():
    r = Registry()
    op = r.resolve("tuple", (V, V))
    batched = torch.zeros(8, 1)
    constant = torch.zeros(1)
    out = {}
    try:
        relaxed(r, op, [batched, constant])
        out["tuple"] = "no error"
    except Exception as exc:
        out["tuple"] = repr(exc)[:120]
    add = r.resolve("add", (V, V), V)
    out["add_shape"] = list(relaxed(r, add, [batched, constant]).shape)
    return out


def warning_3_eq_underflow():
    r = Registry()
    op = r.resolve("eq", (V, V), BOOL)
    rows = []
    first_zero = None
    for d in range(0, 16):
        y = float(relaxed(r, op, [torch.tensor([[float(d)]]), torch.tensor([[0.]])]))
        rows.append({"gap": d, "surrogate": y})
        if y == 0. and first_zero is None:
            first_zero = d
    return {"first_gap_with_exactly_zero_surrogate": first_zero, "rows": rows}


def defect_4_index_is_not_exact_at_integers(n=16, k=8):
    r = Registry()
    arr_t = product(*([V] * n))
    op = r.resolve("index", (arr_t, ADDR), V)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(64, n, generator=g)
    rows = []
    for tau in (1., .5, .25, .145, .1, .05):
        w = torch.softmax(-(torch.tensor([float(k)]) - torch.arange(n)) ** 2 / tau, -1)
        y = relaxed(r, op, [x, torch.tensor([float(k)])], tau)
        rows.append({"temperature": tau,
                     "weight_on_addressed_element": float(w[k]),
                     "max_abs_error_vs_exact": float((y.squeeze(-1) - x[:, k]).abs().max()),
                     "mean_abs_error_vs_exact": float((y.squeeze(-1) - x[:, k]).abs().mean())})
    return {"addresses": n, "reference": k, "shipped_temperature": 1.0, "rows": rows}


def defect_5_one_temperature_two_jobs():
    """The same scalar scales the choice softmax and the operator relaxation."""
    r = Registry()
    arr_t = product(*([V] * 4))
    cands = tuple(Candidate(r.resolve("project", (arr_t,), V, {"index": i}), ("arr",))
                  for i in range(4))
    eq = r.resolve("eq", (V, V), BOOL)
    nodes = (Node("a", V, cands, depth=1),
             Node("h", BOOL, tuple(Candidate(eq, ("a", c)) for c in ("c0", "c1")), depth=2))
    p = Program((("arr", arr_t),), nodes, (("y", "h"),),
                (("c0", Value.of(V, 0.)), ("c1", Value.of(V, 1.)))).validate(r)
    m = SoftProgram(p, r)
    rows = []
    for tau in (1., 100., 10000.):
        m.temperatures["h"] = tau
        logits = torch.tensor([2., 0.])
        m.choices[1].data = logits
        q = torch.softmax(logits / tau, 0)
        x = torch.zeros(4, 4)
        y = relaxed(r, eq, [torch.full((4, 1), 5.), torch.zeros(1)], tau)
        rows.append({"temperature": tau,
                     "choice_distribution": [round(float(v), 6) for v in q],
                     "eq_surrogate_at_gap_5": round(float(y[0, 0]), 8)})
    return {"note": "raising the eq surrogate's temperature also flattens the same "
                    "node's candidate distribution; they share one scalar",
            "rows": rows}


def main():
    out = {
        "warning_1_zero_init": warning_1_zero_init(),
        "warning_2_tuple_broadcast": warning_2_tuple_broadcast(),
        "warning_3_eq_underflow": warning_3_eq_underflow(),
        "defect_4_index_not_exact_at_integers": defect_4_index_is_not_exact_at_integers(),
        "defect_5_one_temperature_two_jobs": defect_5_one_temperature_two_jobs(),
    }
    for k, v in out.items():
        print(f"== {k} ==")
        if "rows" in v:
            for row in v["rows"]:
                print("   ", row)
            print("   ", {kk: vv for kk, vv in v.items() if kk != "rows"})
        else:
            print("   ", v)
    dump("checks", out)


if __name__ == "__main__":
    main()
