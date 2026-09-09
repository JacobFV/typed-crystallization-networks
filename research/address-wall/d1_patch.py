"""Measure the proposed diff D1 itself, rather than an emulation of it.

`ladder_fix.py` reached D1's effect by raising the node temperature, which under
the current `SoftProgram` also flattens that node's candidate softmax -- hence
its "compensated" arm.  That is an emulation, not the diff.  Here the diff is
applied literally, by rebinding `tcn.learning.relaxed` for the duration of this
script only.  NOTHING under `tcn/` is edited; the patched function is a verbatim
copy of the shipped one with the four lines of D1 substituted, and
`checks_patch_is_faithful` asserts the two agree everywhere D1 does not apply.

Run:  .venv/bin/python research/address-wall/d1_patch.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perception-ladder"))

import torch

import tcn.learning as L
from tcn.learning import SoftProgram, tensor
from tcn.operators import Registry
from tcn.search import space_size
from tcn.types import BOOL, floating, integer

from instrument import BUDGET, dump

SHIPPED = L.relaxed
SEEDS = range(12)


def patched(registry, op, xs, temperature=1.):
    """The shipped `relaxed` with D1's `eq` branch substituted."""
    if op.name == "eq" and op.gradient != "none":
        a, b = xs[0], xs[1]
        t = op.inputs[0]
        span = float(2 ** t.bits) if t.kind == "int" and t.encoding.kind == "integer" else 1.
        return torch.exp(-((a - b) ** 2).sum(-1, keepdim=True) / (temperature * span))
    return SHIPPED(registry, op, xs, temperature)


def check_faithful():
    """Everything but `eq` on an integer carrier must be bit-identical."""
    r = Registry()
    F = floating(32)
    bad = []
    for name, types, out in (("add", (F, F), F), ("mul", (F, F), F), ("lt", (F, F), BOOL),
                             ("eq", (F, F), BOOL), ("not", (BOOL,), BOOL)):
        op = r.resolve(name, types, out)
        xs = [torch.tensor([[0.4]]), torch.tensor([[1.7]])][:len(types)]
        a = SHIPPED(r, op, xs); b = patched(r, op, xs)
        if not torch.equal(a, b):
            bad.append(name)
    return {"identical_outside_integer_eq": not bad, "differs_on": bad}


def run_arm(program, registry, inputs, targets, signals, seed, steps=800, lr=.05, noise=.5):
    m = SoftProgram(program, registry)
    g = torch.Generator().manual_seed(seed * 7 + 3)
    with torch.no_grad():
        for p in m.choices:
            if p.requires_grad:
                p.add_(torch.randn(p.shape, generator=g) * noise)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        _, _, trace = m(inputs, return_trace=True)
        BUDGET.add(len(program.nodes))
        m.probe_loss(trace, targets, signals).backward()
        opt.step()
    return m.selections()


def held_out(program, registry, selections, examples, signals):
    worst = 0.
    for e in examples:
        try:
            _, _, tr = program.execute(e["inputs"], registry=registry, selections=selections)
        except Exception:
            return float("inf")
        for s in signals:
            a = tr[s.source].flat(); b = e["targets"][s.target].flat()
            worst = max(worst, max(abs(x - y) for x, y in zip(a, b)))
    return worst


def main():
    t0 = time.perf_counter()
    from rung3_geometry import centre_program, centre_signal, examples as gex
    out = {"faithfulness": check_faithful()}
    print("patch faithful outside integer eq:", out["faithfulness"], flush=True)

    for R in (2, 4):
        r = Registry()
        program = centre_program(r, R, free_address=True)
        signals = centre_signal()
        train = gex(range(0, 24), R); test = gex(range(100, 124), R)
        inputs = {k: torch.stack([tensor(e["inputs"][k]) for e in train]) for k, _ in program.inputs}
        targets = {s.target: torch.stack([tensor(e["targets"][s.target]) for e in train])
                   for s in signals}
        k = (R * R) // 2
        block = {"space": space_size(program), "arms": []}
        print(f"== centre_free_address R={R}: {space_size(program)} programs ==", flush=True)
        for name, fn in (("shipped", SHIPPED), ("D1", patched)):
            L.relaxed = fn
            rows = []
            for seed in SEEDS:
                sel = run_arm(program, r, inputs, targets, signals, seed)
                rows.append({"seed": seed, "selections": sel,
                             "address_is_centre_pixel": sel["hit_px0"] in (3 * k, 3 * k + 1)
                                                        and sel["hit_px1"] in (3 * k, 3 * k + 1),
                             "held_out_max_error": held_out(program, r, sel, test, signals)})
            L.relaxed = SHIPPED
            exact = sum(x["held_out_max_error"] <= 1e-3 for x in rows)
            addr = sum(x["address_is_centre_pixel"] for x in rows)
            block["arms"].append({"arm": name, "held_out_exact": exact,
                                  "centre_pixel_addresses": addr, "n": len(rows), "rows": rows})
            print(f"  {name:9s} held-out exact {exact:2d}/12   "
                  f"both addresses on the centre pixel {addr:2d}/12", flush=True)
        out[f"R{R}"] = block

    BUDGET.stop()
    out["budget"] = BUDGET.to_dict()
    out["wall_seconds"] = round(time.perf_counter() - t0, 2)
    dump("d1_patch", out)


if __name__ == "__main__":
    main()
