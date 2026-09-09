"""Does the fix carry to the recorded arm -- `rung3:centre_free_address`, 576 programs?

That arm is the one FINDINGS section 11 records at 1/12.  It has BOTH the
addresses and the colour constants free, so it exercises the second defect as
well: `SoftProgram` divides a node's candidate softmax and its operator's
relaxation by the same scalar, so scaling `eq`'s surrogate to the byte
representation also flattens the colour choice at that node.

Three arms, all with explicit initialisation noise (`SoftProgram` zero-
initialises logits, so seeds alone vary nothing):

  tau_1                   the shipped behaviour
  tau_256                 the surrogate scaled to int[8]'s span, with the
                          choice-flattening side effect left in
  tau_256_compensated     the same, with the flattened nodes' learning rate
                          multiplied by tau -- what separating the two
                          temperatures (proposed diff D2) would buy

Held-out exact error is reported for every success.

Run:  .venv/bin/python research/address-wall/ladder_fix.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perception-ladder"))

import torch

from tcn.learning import SoftProgram, tensor
from tcn.operators import Registry
from tcn.search import space_size

from instrument import BUDGET, dump

SEEDS = range(12)
STEPS = 800


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


def run(R, tau, compensate, seed, program, registry, inputs, targets, signals,
        steps=STEPS, lr=.05, noise=.5):
    m = SoftProgram(program, registry)
    eqs = [n.name for n in program.nodes if "_eq" in n.name]
    for name in eqs:
        m.temperatures[name] = tau
    g = torch.Generator().manual_seed(seed * 7 + 3)
    with torch.no_grad():
        for p in m.choices:
            if p.requires_grad:
                p.add_(torch.randn(p.shape, generator=g) * noise)
    idx = {n.name: i for i, n in enumerate(program.nodes)}
    hot = [m.choices[idx[n]] for n in eqs]
    cold = [p for i, p in enumerate(m.choices) if p is not None and
            program.nodes[i].name not in eqs and p.requires_grad]
    groups = ([{"params": cold, "lr": lr}, {"params": hot, "lr": lr * (tau if compensate else 1.)}]
              if hot else [{"params": cold, "lr": lr}])
    opt = torch.optim.Adam(groups)
    for _ in range(steps):
        opt.zero_grad()
        _, _, trace = m(inputs, return_trace=True)
        BUDGET.add(len(program.nodes))
        m.probe_loss(trace, targets, signals).backward()
        opt.step()
    return m.selections()


def main():
    t0 = time.perf_counter()
    from rung3_geometry import centre_program, centre_signal, examples as gex, BG
    out = {}
    for R in (2, 4):
        r = Registry()
        program = centre_program(r, R, free_address=True)
        signals = centre_signal()
        train = gex(range(0, 24), R)
        test = gex(range(100, 124), R)
        inputs = {k: torch.stack([tensor(e["inputs"][k]) for e in train]) for k, _ in program.inputs}
        targets = {s.target: torch.stack([tensor(e["targets"][s.target]) for e in train])
                   for s in signals}
        k = (R * R) // 2
        ref = {"bytes": 0, "hit_px0": 3 * k, "hit_px1": 3 * k + 1,
               "hit_eq0": 0, "hit_eq1": 1, "hit": 0}
        block = {"space": space_size(program), "reference": ref, "arms": []}
        print(f"== centre_free_address R={R}: {space_size(program)} programs ==", flush=True)
        for name, tau, comp in (("tau_1", 1., False), ("tau_256", 256., False),
                                ("tau_256_compensated", 256., True)):
            rows = []
            for seed in SEEDS:
                sel = run(R, tau, comp, seed, program, r, inputs, targets, signals)
                hit = sel["hit_px0"] == ref["hit_px0"] and sel["hit_px1"] == ref["hit_px1"]
                rows.append({"seed": seed, "selections": sel, "address_hit": bool(hit),
                             "held_out_max_error": held_out(program, r, sel, test, signals)})
            solved = sum(x["held_out_max_error"] <= 1e-3 for x in rows)
            block["arms"].append({"arm": name, "tau": tau, "compensated": comp,
                                  "address_hits": sum(x["address_hit"] for x in rows),
                                  "held_out_exact": solved, "n": len(rows), "rows": rows})
            print(f"  {name:22s} addresses {sum(x['address_hit'] for x in rows):2d}/12   "
                  f"held-out exact {solved:2d}/12", flush=True)
        out[f"R{R}"] = block
    BUDGET.stop()
    out["budget"] = BUDGET.to_dict()
    out["wall_seconds"] = round(time.perf_counter() - t0, 2)
    dump("ladder_fix", out)


if __name__ == "__main__":
    main()
