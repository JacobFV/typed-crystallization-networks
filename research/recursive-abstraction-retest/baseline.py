"""The chance rate of "module on the output path", so the observed rate means something.

Track 5 warned about exactly this for its own "module selected somewhere" row:
64 of its 68 call-slot candidates were module calls, so an untrained argmax
landed on one with 94% probability and the figure carried no information. The
same trap applies to the output-path measurement, and in the opposite direction
from track 5's scaffolds: here a module is a candidate at *every* node, so a
uniformly random selection puts one on the output path most of the time.

This measures that baseline by sampling uniform random selections in each
scaffold and applying the same `prune` + `module_on_output_path` used on the real
runs. A run only says something about reuse when it beats this, or when it is a
*success* (a module inside a program that is exactly correct is reuse whatever
the prior).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from tcn.operators import Registry

import common as C

HERE = Path(__file__).parent


def sample(prog, registry, trials=20000, seed=0):
    rng = random.Random(seed)
    names = [n.name for n in prog.nodes]
    counts = [len(n.candidates) for n in prog.nodes]
    on_path = 0
    somewhere = 0
    live = 0
    for _ in range(trials):
        sel = {k: rng.randrange(c) for k, c in zip(names, counts)}
        hard = prog.harden(sel)
        p = C.prune(hard)
        live += len(p.nodes)
        if any(n.candidates[n.selected or 0].operator.name.startswith("module:") for n in p.nodes):
            on_path += 1
        if any(n.candidates[sel[n.name]].operator.name.startswith("module:") for n in prog.nodes):
            somewhere += 1
    return {"trials": trials,
            "module_on_output_path_rate": round(on_path / trials, 4),
            "module_selected_somewhere_rate": round(somewhere / trials, 4),
            "mean_live_nodes": round(live / trials, 2)}


def main():
    out = {}
    for label, build in (("wide", C.wide_scaffold), ("tight", C.tight_scaffold)):
        for arm, kind in (("B", "maj"), ("C", "distractor")):
            r = Registry()
            name = r.register_module(C.minimal_module(r, kind))
            prog = build(r, name)
            out[f"{label}_{arm}"] = sample(prog, r)
            out[f"{label}_{arm}"]["candidates_per_node"] = [len(n.candidates) for n in prog.nodes]
    # the chance of a *correct* program under uniform random selection
    out["tight_B_solution_density"] = {"solutions": 144, "space": 2709504,
                                       "rate": 144 / 2709504}
    (HERE / "baseline.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
