"""A/B the block trial: identical seeds, `close_block` off then on.

    .venv/bin/python research/stranded-block/run_ab.py > results/ab.jsonl

Arm A is `close_block=False`, which is `main`'s behaviour exactly.  Arm B is the
fix.  `selections` is recorded for both so the identical-output assertion can be
made on the committed program, not on a summary statistic.  Two further arms run
the stranded seed at rounds 48 and 96 with the fix off, to separate "the residual
is stranded" from "the run merely ran out of rounds".
"""
from __future__ import annotations
import json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from examples.joint import trainer
from tcn.crystallize import Crystallizer, Objective

def run(seed, close_block, episodes=40, rounds=24):
    torch.set_num_threads(1)
    t = trainer(episodes, seed=seed)
    t.run(None)
    sch = Crystallizer(t.model, t.optimizer, tolerance=.05, close_block=close_block)
    def validation(regularized=True):
        return sum(t.episode(20000+i, False, 'validation', loss_only=True,
                             regularized=regularized) for i in range(2)) / 2
    sch.run(Objective(validation, lambda: validation(False)), rounds=rounds, retrain_steps=2)
    total = len(t.model.program.nodes); frozen = len(t.model.frozen)
    return {"seed": seed, "close_block": close_block, "rounds": rounds,
            "frozen": frozen, "nodes": total, "fully_frozen": frozen == total,
            "selections": {k: v for k, v in sorted(t.model.selections().items())},
            "residual": sorted(n.name for n in t.model.program.nodes if n.name not in t.model.frozen),
            "disconnected_refusals": sum(1 for e in sch.events if e.reason == "disconnected remaining region"),
            "node_trials": sum(1 for e in sch.events if not e.reason.startswith("block")),
            "block_events": [vars(e) for e in sch.block_events]}

if __name__ == "__main__":
    for s in range(8):
        for cb in (False, True):
            print(json.dumps(run(s, cb)), flush=True)
    # Is the residual stranded, or did the run merely run out of rounds?
    for rounds in (48, 96):
        print(json.dumps(run(7, False, rounds=rounds)), flush=True)
