"""Does the perturbation measurement itself agree with argmax, before any freezing?

The joint tables separate two things that the scheduler conflates: *what the
measurement says* and *when the scheduler acts on it*. This script isolates the
first. It trains `examples/joint.py` for N episodes and then, with nothing
frozen and nothing retrained, reports for each real decision node the argmax
candidate, the perturbation-selected candidate, and the decisiveness gap.

The reference program is relation = goal_relation = truth table 6.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from examples.joint import trainer  # noqa: E402
from tcn.crystallize import Crystallizer  # noqa: E402


def probe(episodes, seed):
    torch.set_num_threads(1)
    t = trainer(episodes, seed=seed)
    t.run(None)

    def validation():
        return sum(t.episode(20000 + i, False, "validation", loss_only=True) for i in range(2)) / 2

    scheduler = Crystallizer(t.model, t.optimizer)
    row = {"episodes": episodes, "seed": seed, "nodes": {}}
    for index, node in enumerate(t.model.program.nodes):
        if len(node.candidates) == 1:
            continue
        scores = scheduler.perturbation_scores(index, node, validation)
        order = sorted(range(len(scores)), key=lambda k: -scores[k])
        row["nodes"][node.name] = {
            "argmax": int(t.model.choices[index].argmax()),
            "perturbation": order[0],
            "gap": scores[order[0]] - scores[order[1]],
            "scores": scores,
        }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", default=[40, 80, 160])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "selection-probe.json"))
    args = ap.parse_args()
    rows = [probe(e, s) for e in args.episodes for s in range(args.seeds)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"{'episodes':>8} {'seed':>4} {'node':16} {'argmax':>7} {'perturb':>8} {'gap':>12}")
    for r in rows:
        for name, d in r["nodes"].items():
            print(f"{r['episodes']:8} {r['seed']:4} {name:16} {d['argmax']:7} {d['perturbation']:8} {d['gap']:12.4g}")
    for e in args.episodes:
        sub = [r for r in rows if r["episodes"] == e]
        agree = sum(1 for r in sub for d in r["nodes"].values() if d["argmax"] == d["perturbation"])
        total = sum(len(r["nodes"]) for r in sub)
        correct_a = sum(1 for r in sub for d in r["nodes"].values() if d["argmax"] == 6)
        correct_p = sum(1 for r in sub for d in r["nodes"].values() if d["perturbation"] == 6)
        print(f"\nepisodes {e}: agree {agree}/{total}; argmax picks table 6 in {correct_a}/{total}; "
              f"perturbation picks table 6 in {correct_p}/{total}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
