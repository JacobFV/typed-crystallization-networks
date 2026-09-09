"""Per-node autopsy of the arm that actually fails: `rung3:centre_free_address`.

`pointing.py` reports one aggregate number per scaffold.  That aggregate mixes
node roles, and it silently drops every node whose gradient is exactly zero, so
"picks the reference 0.25 of the time" cannot distinguish a misleading gradient
from an absent one.  This splits the same measurement by node role and reports
the dropped nodes as data.

Contrasted against the arm that succeeds, `rung3:mask_pinned_address`, on the
same episodes.

Run:  .venv/bin/python research/address-wall/ladder_autopsy.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perception-ladder"))

import torch

from tcn.learning import SoftProgram, tensor
from tcn.operators import Registry

from instrument import BUDGET, dump

ROLE = {"px": "address", "eq": "colour", "hit": "logic", "acc": "logic", "bytes": "fixed"}


def role_of(name):
    if "_px" in name:
        return "address"
    if "_eq" in name:
        return "colour"
    if name.startswith("acc") or name == "hit" or name.startswith("hit") and "_" not in name:
        return "logic"
    return "other"


def autopsy(program, examples, signals, registry, reference, seeds=8, noise=.5, label=""):
    inputs = {k: torch.stack([tensor(e["inputs"][k]) for e in examples]) for k, _ in program.inputs}
    targets = {s.target: torch.stack([tensor(e["targets"][s.target]) for e in examples]) for s in signals}
    per_role = {}
    forwards = {}
    for seed in range(seeds):
        m = SoftProgram(program, registry)
        g = torch.Generator().manual_seed(seed + 999)
        with torch.no_grad():
            for p in m.choices:
                p.add_(torch.randn(p.shape, generator=g) * noise)
        _, _, trace = m(inputs, return_trace=True)
        BUDGET.add(len(program.nodes))
        loss = m.probe_loss(trace, targets, signals)
        loss.backward()
        if seed == 0:
            forwards = {k: [round(float(v.detach().reshape(-1)[0]), 6),
                            round(float(v.detach().float().mean()), 6)]
                        for k, v in trace.items() if v.numel() < 10000}
        for node, p in zip(program.nodes, m.choices):
            if len(node.candidates) < 2:
                continue
            r = role_of(node.name)
            d = per_role.setdefault(r, {"nodes": 0, "dead": 0, "scored": 0, "hits": 0,
                                        "chance": 0., "grad_abs_sum": []})
            d["nodes"] += 1
            d["chance"] += 1 / len(node.candidates)
            if p.grad is None or float(p.grad.abs().sum()) == 0.:
                d["dead"] += 1
                d["grad_abs_sum"].append(0.)
                continue
            d["scored"] += 1
            d["grad_abs_sum"].append(float(p.grad.abs().sum()))
            d["hits"] += int(int(p.grad.argmin()) == reference[node.name])
    out = {}
    for r, d in per_role.items():
        out[r] = {"nodes": d["nodes"], "dead_gradient": d["dead"], "scored": d["scored"],
                  "picks_reference": d["hits"] / max(1, d["scored"]),
                  "chance": d["chance"] / max(1, d["nodes"]),
                  "median_grad_abs_sum": float(torch.tensor(d["grad_abs_sum"]).median()),
                  "dead_fraction": d["dead"] / max(1, d["nodes"])}
        print(f"  [{label}] {r:9s} nodes={d['nodes']:3d} dead={d['dead']:3d} "
              f"scored={d['scored']:3d} picks-ref={out[r]['picks_reference']:.3f} "
              f"chance={out[r]['chance']:.3f} |grad|={out[r]['median_grad_abs_sum']:.3g}",
              flush=True)
    return {"per_role": out, "forward_at_init_seed0": forwards}


def main():
    t0 = time.perf_counter()
    from rung3_geometry import (centre_program, mask_program, centre_signal,
                                mask_signals, examples as gex, BG, AND, OR)
    ex = gex(range(0, 24), 2, dense=True)
    out = {}

    print("== free addresses (the failing arm) ==", flush=True)
    r = Registry(); p = centre_program(r, 2, free_address=True)
    out["centre_free_address_R2"] = autopsy(
        p, ex, centre_signal(), r,
        {"bytes": 0, "hit_px0": 6, "hit_px1": 7, "hit_eq0": 0, "hit_eq1": 1, "hit": 0},
        label="free")

    print("== pinned addresses (the succeeding arm) ==", flush=True)
    POOL = (24, 30, 43, 15)
    r = Registry(); p = mask_program(r, 2, free_address=False, pool=POOL, and_menu=(AND, OR))
    ref = {n.name: 0 for n in p.nodes}
    for i in range(4):
        ref[f"hit{i}_eq0"] = POOL.index(BG[0]); ref[f"hit{i}_eq1"] = POOL.index(BG[1]); ref[f"hit{i}"] = 0
    out["mask_pinned_address_R2"] = autopsy(p, ex, mask_signals(2), r, ref, label="pinned")

    # what the `eq` surrogate sees at the uniform address mixture
    r = Registry(); p = centre_program(r, 2, free_address=True)
    m = SoftProgram(p, r)
    inputs = {k: torch.stack([tensor(e["inputs"][k]) for e in ex]) for k, _ in p.inputs}
    _, _, trace = m(inputs, return_trace=True)
    mix = trace["hit_px0"].detach()
    ref_px = trace["bytes"].detach()[:, 6]
    out["mixture_diagnostic"] = {
        "mixed_address_value_mean": float(mix.mean()),
        "reference_address_value_mean": float(ref_px.mean()),
        "eq_constant": BG[0],
        "abs_gap_at_mixture": float((mix.reshape(-1) - BG[0]).abs().mean()),
        "abs_gap_at_reference": float((ref_px - BG[0]).abs().mean()),
        "eq_surrogate_at_mixture": float(torch.exp(-((mix.reshape(-1) - BG[0]) ** 2)).mean()),
        "eq_surrogate_at_reference": float(torch.exp(-((ref_px - BG[0]) ** 2)).mean()),
        "float32_underflow_gap": 11,
    }
    print("== eq surrogate at the address mixture ==", flush=True)
    for k, v in out["mixture_diagnostic"].items():
        print(f"   {k:32s} {v}", flush=True)

    BUDGET.stop()
    out["budget"] = BUDGET.to_dict()
    out["wall_seconds"] = round(time.perf_counter() - t0, 2)
    dump("ladder_autopsy", out)


if __name__ == "__main__":
    main()
