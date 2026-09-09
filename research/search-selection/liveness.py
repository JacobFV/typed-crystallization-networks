"""Is the surrogate alive at the data? Three questions, all measured.

1. **The failing reading itself.** What `eq`'s relaxation and its derivative
   actually are, as a function of |a-b|, in float32 at tau=1 and at tau=2^bits.
   FINDINGS section 16 records the threshold at |a-b| >= 11; that is checked
   here rather than quoted.

2. **What the check says on the arm that produced a wrong headline.** The
   recorded rung-3 free-address scaffold, at R=2 and R=4, before and after
   carrier scaling: which candidates are dead, on how many examples, and what
   share of the declared space gradient descent can reach at all.

3. **That separating the two temperatures changed nothing shipped.** Every
   fixture in the repository is run through `SoftProgram` with the new
   `surrogate_scale` axis at its default and compared, tensor by tensor, with
   the same forward taken through the shipped code path.

Run:  PYTHONPATH=. .venv/bin/python research/search-selection/liveness.py
"""
from __future__ import annotations
import time

import torch

from common import CASES, dump, exact_error
from tcn.learning import SoftProgram, carrier_temperature, relaxed, tensor
from tcn.operators import Registry
from tcn.select import DEAD_GRADIENT, liveness
from tcn.types import BOOL, integer


def eq_profile():
    """`exp(-(a-b)^2/tau)` and its derivative, read off in float32."""
    from tcn.generation import BYTE
    r = Registry()
    op = r.resolve("eq", (BYTE, BYTE))
    rows = []
    for d in list(range(0, 20)) + [24, 25, 26, 32, 64, 128, 255]:
        row = {"spread": d}
        for scaled in (False, True):
            a = torch.tensor([[float(d)]], requires_grad=True)
            b = torch.tensor([[0.]])
            y = relaxed(r, op, [a, b], 1., scaled)
            g, = torch.autograd.grad(y.sum(), [a])
            row["value_tau1" if not scaled else "value_tau256"] = float(y.detach())
            row["grad_tau1" if not scaled else "grad_tau256"] = float(g)
        rows.append(row)
    # d/da exp(-(a-b)^2/tau) is -2(a-b)/tau * exp(...), which is zero AT a == b
    # for the uninteresting reason. The threshold is the smallest positive
    # spread at which it underflows and stays there.
    first_dead = next(x["spread"] for x in rows if x["spread"] > 0 and x["grad_tau1"] == 0.)
    first_zero_value = next(x["spread"] for x in rows if x["value_tau1"] == 0.)
    return {"carrier_temperature_int8": carrier_temperature(BYTE),
            "first_spread_with_exactly_zero_value_at_tau1": first_zero_value,
            "carrier_temperature_bool": carrier_temperature(BOOL),
            "carrier_temperature_int16": carrier_temperature(integer(16, signed=False)),
            "first_spread_with_exactly_zero_gradient_at_tau1": first_dead,
            "rows": rows}


def arm(case):
    out = {"case": case.name, "space": case.space, "arms": []}
    for scaled in (False, True):
        t = time.perf_counter()
        lv = liveness(case.program, case.train, case.signals, case.registry, carrier_scaled=scaled)
        out["arms"].append({
            "carrier_scaled": scaled,
            "seconds": round(time.perf_counter() - t, 3),
            "reachable_fraction": lv.reachable_fraction,
            "dead_nodes": list(lv.dead_nodes),
            "blind_nodes": list(lv.blind_nodes),
            "boundary_nodes": list(lv.boundary_nodes),
            "node_signal": {k: float(v) for k, v in lv.node_signal.items()},
            "constant_signal": {k: float(v) for k, v in lv.constant_signal.items()},
            "live_constants": list(lv.live_constants),
            "eq_spread": {k: v for k, v in lv.eq_spread.items()},
            "dead_candidates": [{"node": c.node, "index": c.index, "operator": c.operator,
                                 "live_rows": c.live_rows, "rows": c.rows,
                                 "peak_gradient": c.sensitivity, "detail": c.detail}
                                for c in lv.candidates if c.dead],
            "live_row_histogram": sorted({(c.node, c.index): c.live_rows for c in lv.candidates}.values()),
        })
    return out


def fixtures_unchanged():
    """The `surrogate_scale` axis at its default must be bit-identical."""
    import tcn.learning as L
    shipped = L.relaxed
    def one_argument(registry, op, xs, temperature=1., carrier_scaled=False):
        """The relaxation with the new keyword removed, i.e. the pre-change contract."""
        assert carrier_scaled is False, "default path must never request carrier scaling"
        return shipped(registry, op, xs, temperature)
    results = []
    for name in ("mixed", "mixed_constant", "joint_offline", "rung3_R2", "noisy_2flip", "gradient_boundary"):
        case = CASES[name]()
        m = SoftProgram(case.program, case.registry)
        inputs = {k: torch.stack([tensor(e["inputs"][k]) for e in case.train]) for k, _ in case.program.inputs}
        assert all(v == 1. for v in m.surrogate_scale.values())
        assert m.carrier_scaled is False
        a, _, _ = m(inputs, return_trace=True)
        L.relaxed = one_argument
        try:
            b, _, _ = SoftProgram(case.program, case.registry)(inputs, return_trace=True)
        finally:
            L.relaxed = shipped
        same = all(torch.equal(a[k], b[k]) for k in a)
        results.append({"fixture": name, "outputs": sorted(a), "bit_identical": bool(same)})
    return results


def main():
    t0 = time.perf_counter()
    out = {"dead_gradient_threshold": DEAD_GRADIENT, "eq_profile": eq_profile(),
           "shipped_forward_bit_identical": fixtures_unchanged(), "cases": []}
    print("eq at tau=1 first has an exactly-zero gradient at |a-b| =",
          out["eq_profile"]["first_spread_with_exactly_zero_gradient_at_tau1"], flush=True)
    for row in out["shipped_forward_bit_identical"]:
        print(f"  shipped forward unchanged on {row['fixture']:18s} {row['bit_identical']}", flush=True)
    for name in ("mixed", "rung3_R2", "rung3_R4", "rung3_bytes", "gradient_boundary", "boundary_wide",
                 "joint_offline", "noisy_2flip", "wide_logic", "mixed_constant"):
        case = CASES[name]()
        block = arm(case)
        out["cases"].append(block)
        for a in block["arms"]:
            print(f"  {name:18s} scaled={str(a['carrier_scaled']):5s} "
                  f"reachable={a['reachable_fraction']:.4g} dead={a['dead_nodes']} "
                  f"blind={a['blind_nodes']} dead_candidates={len(a['dead_candidates'])} "
                  f"live_constants={a['live_constants']}", flush=True)
    out["wall_seconds"] = round(time.perf_counter() - t0, 2)
    dump("liveness", out)


if __name__ == "__main__":
    main()
