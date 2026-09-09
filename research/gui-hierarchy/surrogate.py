"""Is the gradient into the address choice small, or identically zero?

Conflating those two produced a wrong finding once already, so this track reports
the number that separates them.  `eq`'s training relaxation is
`exp(-(a-b)^2 / tau)` with the shipped `tau = 1`, which is exactly 0.0 in float32
for `|a-b| >= 11`.  Every choice *upstream* of an `eq` -- here the `shifted` node,
which is an address choice -- receives its gradient through that surrogate, so if
the surrogate is dead at the operating distance the address logits get exactly
zero and no amount of search budget changes it.

Measured here:

1. the palette's own separations, and the surrogate at each, at the shipped
   `tau = 1` and at the proposed carrier-width `tau = 2^bits = 256`;
2. the distribution of `|a-b|` actually seen at the rung-1 operating point;
3. the real gradient magnitude at each choice node of the rung-1 scaffold at
   initialisation, so "the address logits are dead" is an observation rather than
   an inference.

One caveat worth stating because it constrains any fix: `SoftProgram` keeps ONE
temperature per node and uses it for both the candidate softmax and the operator
relaxation, so widening this surrogate also flattens that node's choice
distribution.  They are not independently tunable on current main.
"""
from __future__ import annotations

import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import colour_at, dump, edge_positions, episode, report
from rung1_edges import SCREEN, examples, narrow_offsets, scaffold, signals, wide_offsets
from tcn.learning import SoftProgram, tensor as _tensor
from tcn.operators import Registry
from generators.gui.render import palette_levels


def surrogate(distance, tau):
    return float(torch.exp(torch.tensor(-(distance ** 2) / tau, dtype=torch.float32)))


def slope(distance, tau):
    """|d/dd exp(-d^2/tau)| -- the quantity a choice upstream of `eq` actually gets.

    It vanishes at BOTH ends: exactly 0 at d = 0 (the peak) and exactly 0 once the
    exponential underflows.  Reporting the value alone hides the first of those.
    """
    return abs(2 * distance / tau) * surrogate(distance, tau)


def palette_table():
    rows = {}
    for levels in (2, 4, 8, 16, 32):
        values = palette_levels(levels)
        separation = values[1] - values[0]
        rows[levels] = {"colours": levels ** 3, "separation": separation,
                        "surrogate_tau_1": surrogate(separation, 1.),
                        "surrogate_tau_256": surrogate(separation, 256.),
                        "slope_tau_1": slope(separation, 1.),
                        "slope_tau_256": slope(separation, 256.),
                        "live_at_tau_1": slope(separation, 1.) > 1e-6,
                        "live_at_tau_256": slope(separation, 256.) > 1e-6}
        report(f"palette_levels={levels}",
               f"colours {levels ** 3:6d} separation {separation:3d} "
               f"eq tau=1 value {rows[levels]['surrogate_tau_1']:.2e} slope "
               f"{rows[levels]['slope_tau_1']:.2e} | tau=256 value "
               f"{rows[levels]['surrogate_tau_256']:.2e} slope {rows[levels]['slope_tau_256']:.2e}")
    return rows


def operating_distances(seeds, **configuration):
    """|a-b| per channel between a pixel and its right neighbour, over real screens."""
    counts = {}
    for seed in seeds:
        ep = episode(seed, "train", **configuration)
        for i in edge_positions(ep):
            a, b = colour_at(ep, i), colour_at(ep, i + 1)
            for x, y in zip(a, b):
                counts[abs(x - y)] = counts.get(abs(x - y), 0) + 1
    total = sum(counts.values())
    return {"histogram": {str(k): v for k, v in sorted(counts.items())},
            "fraction_zero": counts.get(0, 0) / total,
            "fraction_live_tau_1": sum(v for k, v in counts.items()
                                       if slope(k, 1.) > 1e-6) / total,
            "fraction_live_tau_256": sum(v for k, v in counts.items()
                                         if slope(k, 256.) > 1e-6) / total,
            "distinct_distances": sorted(counts)}


def choice_gradients(registry, program, rows, label):  # noqa: D401
    """Gradient magnitude at every choice node after one backward pass."""
    model = SoftProgram(program, registry)
    inputs = {k: torch.stack([_tensor(ex["inputs"][k]) for ex in rows])
              for k, _ in program.inputs}
    targets = {s.target: torch.stack([_tensor(ex["targets"][s.target]) for ex in rows])
               for s in signals()}
    _, _, trace = model(inputs, return_trace=True)
    loss = model.probe_loss(trace, targets, signals())
    loss.backward()
    out = {}
    for node, parameter in zip(program.nodes, model.choices):
        if len(node.candidates) < 2:
            continue
        grad = parameter.grad
        out[node.name] = {"candidates": len(node.candidates),
                          "grad_is_none": grad is None,
                          "grad_abs_max": 0. if grad is None else float(grad.abs().max()),
                          "grad_norm": 0. if grad is None else float(grad.norm())}
        report(f"[{label}] {node.name} ({len(node.candidates)} candidates)",
               f"grad_is_none={out[node.name]['grad_is_none']} "
               f"|g|max={out[node.name]['grad_abs_max']:.3e}")
    out["_loss"] = float(loss)
    return out


def main():
    registry = Registry()
    result = {"screen": SCREEN}
    print("--- palette separation against the eq surrogate ---")
    result["palette"] = palette_table()

    print("\n--- distances actually seen at the rung-1 operating point ---")
    for levels in (4, 16, 32):
        configuration = dict(SCREEN, palette_levels=levels,
                             palette=min(32, levels ** 3))
        row = operating_distances(range(4), **configuration)
        result[f"operating_levels_{levels}"] = row
        report(f"palette_levels={levels} neighbour |a-b|",
               f"zero {row['fraction_zero']:.4f}  live tau=1 {row['fraction_live_tau_1']:.4f}  "
               f"live tau=256 {row['fraction_live_tau_256']:.4f}  "
               f"distances {row['distinct_distances'][:8]}")

    print("\n--- gradient at each choice node, one backward pass at initialisation ---")
    for levels in (4, 16):
        configuration = dict(SCREEN, palette_levels=levels, palette=min(32, levels ** 3))
        probe = episode(0, "train", **configuration)
        rows = examples(range(4), "train", 24, seed=1, **configuration)
        for pool, tag in ((narrow_offsets(probe["width"]), "narrow"),
                          (wide_offsets(probe["width"]), "wide")):
            for relax, mode in ((False, "single-candidate nodes pre-selected"),
                                (True, "single-candidate nodes relaxed")):
                program = scaffold(registry, probe, pool, False, relax_single=relax)
                key = f"gradients_levels_{levels}_{tag}_{'relaxed' if relax else 'preselected'}"
                result[key] = choice_gradients(registry, program, rows,
                                               f"levels={levels} {tag} | {mode}")
    dump("surrogate", result)


if __name__ == "__main__":
    main()
