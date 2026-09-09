"""Re-run the two benchmarks the defects were measured on, before and after.

The carrier scaling is opt-in, so `before` is the shipped default with nothing
turned on, and `after` is the same run with `SoftProgram.scale_surrogates()`
called -- the whole difference between the arms, and the only thing the caller
does differently.

MEASUREMENT WARNING. `SoftProgram` zero-initializes every choice logit, so
`torch.manual_seed` alone does not vary synthesis at all. Every arm here adds
explicit Gaussian noise to the trainable logits (`noise=0.5`), which is what
makes a seed mean anything; the seeds are the same in both arms.

Run:  .venv/bin/python research/core-gradient-fixes/benchmarks.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perception-ladder"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "address-wall"))

import torch

from tcn.learning import SoftProgram, tensor
from tcn.operators import Registry
from tcn.search import space_size

SEEDS = range(12)
EVALUATIONS = [0]


def run_arm(program, registry, inputs, targets, signals, seed, scaled, steps=800, lr=.05, noise=.5):
    m = SoftProgram(program, registry)
    if scaled: m.scale_surrogates()
    g = torch.Generator().manual_seed(seed * 7 + 3)
    with torch.no_grad():
        for p in m.choices:
            if p.requires_grad:
                p.add_(torch.randn(p.shape, generator=g) * noise)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        _, _, trace = m(inputs, return_trace=True)
        EVALUATIONS[0] += len(program.nodes)
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


def centre_free_address():
    """`rung3:centre_free_address` -- the arm FINDINGS section 16 records at 0/12."""
    from rung3_geometry import centre_program, centre_signal, examples as gex
    out = {}
    for R in (2, 4):
        r = Registry()
        program = centre_program(r, R, free_address=True)
        signals = centre_signal()
        train = gex(range(0, 24), R); test = gex(range(100, 124), R)
        inputs = {k: torch.stack([tensor(e["inputs"][k]) for e in train]) for k, _ in program.inputs}
        targets = {s.target: torch.stack([tensor(e["targets"][s.target]) for e in train]) for s in signals}
        k = (R * R) // 2
        block = {"space": space_size(program), "arms": {}}
        print(f"== centre_free_address R={R}: {space_size(program)} programs ==", flush=True)
        for name, scaled in (("before", False), ("after", True)):
            rows = []
            for seed in SEEDS:
                sel = run_arm(program, r, inputs, targets, signals, seed, scaled)
                rows.append({"seed": seed,
                             "address_is_centre_pixel": sel["hit_px0"] in (3 * k, 3 * k + 1)
                                                        and sel["hit_px1"] in (3 * k, 3 * k + 1),
                             "held_out_max_error": held_out(program, r, sel, test, signals)})
            block["arms"][name] = {
                "held_out_exact": sum(x["held_out_max_error"] <= 1e-3 for x in rows),
                "centre_pixel_addresses": sum(x["address_is_centre_pixel"] for x in rows),
                "n": len(rows), "rows": rows}
            a = block["arms"][name]
            print(f"  {name:7s} held-out exact {a['held_out_exact']:2d}/12   "
                  f"both addresses on the centre pixel {a['centre_pixel_addresses']:2d}/12", flush=True)
        out[f"R{R}"] = block
    return out


if __name__ == "__main__":
    t0 = time.perf_counter()
    torch.set_num_threads(1)
    report = {"centre_free_address": centre_free_address()}
    report["node_evaluations"] = EVALUATIONS[0]
    report["wall_seconds"] = round(time.perf_counter() - t0, 2)
    Path(__file__).with_name("benchmarks.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"node_evaluations": report["node_evaluations"],
                      "wall_seconds": report["wall_seconds"]}))
