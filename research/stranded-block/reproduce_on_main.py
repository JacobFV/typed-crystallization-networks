"""Reproduce the stranded-residual hazard on `main`, with no seasons machinery.

The shipped `Crystallizer`, the shipped `examples/joint` fixture, and exactly the
settings `tcn.cli.joint` uses: 40 training episodes, rounds=24, retrain_steps=2,
tolerance=0.05.  Prints one JSON record per seed.

    .venv/bin/python research/stranded-block/reproduce_on_main.py 8

A seed with `fully_frozen: false` and a large `disconnected_refusals` is the
hazard: the per-node viability guard refusing the same residual set every round.
"""
from __future__ import annotations
import json, sys, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from examples.joint import trainer
from tcn.crystallize import Crystallizer, Objective

def run(seed, episodes=40):
    torch.set_num_threads(1)
    t = trainer(episodes, seed=seed)
    t.run(None)
    sch = Crystallizer(t.model, t.optimizer, tolerance=.05)
    def validation(regularized=True):
        return sum(t.episode(20000+i, False, 'validation', loss_only=True,
                             regularized=regularized) for i in range(2)) / 2
    sch.run(Objective(validation, lambda: validation(False)), rounds=24, retrain_steps=2)
    total = len(t.model.program.nodes)
    frozen = len(t.model.frozen)
    reasons = collections.Counter(e.reason for e in sch.events)
    residual = sorted(n.name for n in t.model.program.nodes if n.name not in t.model.frozen)
    disc = [e.node for e in sch.events if e.reason == "disconnected remaining region"]
    return {"seed": seed, "nodes": total, "frozen": frozen,
            "fully_frozen": frozen == total, "reasons": dict(reasons),
            "residual": residual,
            "disconnected_refusals": len(disc),
            "disconnected_names": sorted(set(disc))}

if __name__ == "__main__":
    for s in range(int(sys.argv[1]) if len(sys.argv) > 1 else 8):
        print(json.dumps(run(s)), flush=True)
