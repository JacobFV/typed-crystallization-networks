"""At initialisation, does the loss gradient point at the reference program?

For each free node the Adam step raises the weight of whichever candidate has
the most negative logit gradient.  If that is the reference candidate, the
relaxation is giving usable local information about the discrete answer.
Compared against the chance rate 1/|candidates|.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from tcn.operators import Registry
from tcn.learning import SoftProgram, tensor
from common import dump


def score(program, examples, signals, registry, reference, noise=.5, seeds=8):
    inputs = {k: torch.stack([tensor(e["inputs"][k]) for e in examples]) for k, _ in program.inputs}
    targets = {s.target: torch.stack([tensor(e["targets"][s.target]) for e in examples]) for s in signals}
    hits = total = 0; chance = 0.
    for seed in range(seeds):
        m = SoftProgram(program, registry)
        g = torch.Generator().manual_seed(seed + 999)
        with torch.no_grad():
            for p in m.choices: p.add_(torch.randn(p.shape, generator=g) * noise)
        try:
            _, _, trace = m(inputs, return_trace=True)
            loss = m.probe_loss(trace, targets, signals); loss.backward()
        except Exception:
            continue
        for node, p in zip(program.nodes, m.choices):
            if len(node.candidates) < 2 or p.grad is None or float(p.grad.abs().sum()) == 0.: continue
            total += 1; chance += 1 / len(node.candidates)
            hits += int(int(p.grad.argmin()) == reference[node.name])
    return {"nodes_scored": total, "points_at_reference": hits / max(1, total),
            "chance": chance / max(1, total)}


if __name__ == "__main__":
    out = {}
    from rung1_signal import frequency_program, sample_examples, LADDER, FREQ_SIGNAL, WINDOW
    ex = sample_examples(range(0, 60))[:10]
    for name, free in LADDER.items():
        r = Registry(); p = frequency_program(r, free=free)
        ref = {n.name: 0 for n in p.nodes}
        for k, v in (("a", WINDOW - 1), ("b", WINDOW - 2), ("d", WINDOW - 3)):
            if k in free: ref[k] = v
        out[f"rung1:{name}"] = score(p, ex, (FREQ_SIGNAL,), r, ref)

    from rung3_geometry import (centre_program, mask_program, centre_signal, mask_signals,
                                examples as gex, BG, AND, OR)
    g2 = gex(range(0, 24), 2, dense=True)
    r = Registry(); p = centre_program(r, 2, free_address=True)
    out["rung3:centre_free_address_R2"] = score(
        p, g2, centre_signal(), r, {"bytes": 0, "hit_px0": 6, "hit_px1": 7, "hit_eq0": 0, "hit_eq1": 1, "hit": 0})
    r = Registry(); pool = tuple(range(256)); p = centre_program(r, 2, free_address=False, pool=pool)
    out["rung3:centre_colour_search_R2"] = score(
        p, g2, centre_signal(), r,
        {"bytes": 0, "hit_px0": 0, "hit_px1": 0, "hit_eq0": pool.index(BG[0]), "hit_eq1": pool.index(BG[1]), "hit": 0})
    POOL = (24, 30, 43, 15)
    r = Registry(); p = mask_program(r, 2, free_address=False, pool=POOL, and_menu=(AND, OR))
    ref = {n.name: 0 for n in p.nodes}
    for i in range(4):
        ref[f"hit{i}_eq0"] = POOL.index(BG[0]); ref[f"hit{i}_eq1"] = POOL.index(BG[1]); ref[f"hit{i}"] = 0
    out["rung3:mask_pinned_address_R2"] = score(p, g2, mask_signals(2), r, ref)

    dump("pointing", out)
    for k, v in out.items():
        print(f"{k:36s} nodes={v['nodes_scored']:4d} gradient picks the reference "
              f"{v['points_at_reference']:.3f} (chance {v['chance']:.3f})")
