"""What the connectivity guard sees, parameter by parameter, on the joint fixture.

Trains `examples/joint.py`, then for every node in turn: freeze it, and record for
each still-trainable parameter whether the gradient of

  * the regularized total  (prediction + probe + actor + value + policy entropy
    + crystal_weight * model.entropy() + mdl_weight * model.complexity())
  * the unregularized task loss (the same without the last two terms)

is absent (`None`, i.e. no path in the autograd graph) or present-but-zero. Three
rules are then evaluated against the same measurement:

  reachability(total)  the shipped guard
  reachability(task)   the guard after this change
  nonzero(task)        the "all-zero gradient means disconnected" rule that was
                       implemented and reverted

The freeze is rolled back after each node, so every row is measured against the
same trained model.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from examples.joint import trainer  # noqa: E402


def probe(episodes=160, seed=0):
    torch.set_num_threads(1)
    t = trainer(episodes, seed=seed)
    t.run(None)
    model = t.model

    def objective(regularized):
        return sum(t.episode(20000 + i, False, "validation", loss_only=True, regularized=regularized)
                   for i in range(2)) / 2

    rows = []
    for node in model.program.nodes:
        if node.name in model.frozen:
            continue
        weights = copy.deepcopy(model.state_dict())
        frozen = dict(model.frozen)
        requires = [p.requires_grad for p in model.parameters()]
        model.freeze(node.name, model.selections()[node.name])
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        active = [p for p in model.parameters() if p.requires_grad]
        row = {"node": node.name, "candidates": len(node.candidates), "active": len(active)}
        for label, regularized in (("total", True), ("task", False)):
            loss = objective(regularized)
            grads = (torch.autograd.grad(loss, active, allow_unused=True)
                     if active and loss.requires_grad else [None] * len(active))
            row[f"{label}_absent"] = [n for n, g in zip(names, grads) if g is None]
            row[f"{label}_zero"] = [n for n, g in zip(names, grads)
                                    if g is not None and not torch.any(g != 0)]
        row["reachability_total_fires"] = bool(row["total_absent"])
        row["reachability_task_fires"] = bool(row["task_absent"])
        row["nonzero_task_fires"] = bool(row["task_absent"] or row["task_zero"])
        rows.append(row)
        model.load_state_dict(weights)
        model.frozen = frozen
        for p, flag in zip(model.parameters(), requires):
            p.requires_grad_(flag)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "guard-probe.json"))
    args = ap.parse_args()
    rows = probe(args.episodes, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"episodes": args.episodes, "seed": args.seed, "nodes": rows}, indent=2))
    print(f"{'node':16} {'cand':>5} {'reach(total)':>13} {'reach(task)':>12} {'nonzero(task)':>14}  absent under task only")
    for r in rows:
        extra = sorted(set(r["task_absent"]) - set(r["total_absent"]))
        print(f"{r['node']:16} {r['candidates']:5} {str(r['reachability_total_fires']):>13} "
              f"{str(r['reachability_task_fires']):>12} {str(r['nonzero_task_fires']):>14}  "
              + ", ".join(extra))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
