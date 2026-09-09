"""The same population, on the environment-coupled `joint` fixture.

Selection rule, ancestry, reseeding and the `--select none` control are exactly
`population.py`'s; only the fixture and the scoring loss differ.  Every member is
a separate `examples.joint.trainer` at the SAME seed -- so the episode stream is
shared and members differ only by the Gaussian noise added to their choice logits
-- and the episode budget is spent in generations by advancing each member's
`config.episodes` and calling `run` again, which the trainer already supports
through `completed`.

Environment cost is the honest total: P members x N episodes of training, plus
every member's crystallization rollouts, summed and reported as one number.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from examples.joint import trainer  # noqa: E402
from tcn.crystallize import Objective  # noqa: E402

from arms import (count_forwards, count_steps, freeze_all_argmax,  # noqa: E402
                  make_scheduler, reason_histogram, rounds_for, scheduler_record)
from population import dominated  # noqa: E402
from run_joint import EPISODES, RETRAIN_STEPS, TOLERANCE, frozen_agent_return  # noqa: E402


class Member:
    def __init__(self, index, seed, episodes, noise):
        self.index = index
        self.trainer = trainer(episodes, seed=seed)
        self.trainer.config.episodes = 0
        if noise:
            with torch.no_grad():
                for logits in self.trainer.model.choices:
                    logits.add_(torch.randn(logits.shape) * noise)
        self.steps = count_steps(self.trainer.optimizer)
        self.forwards = count_forwards(self.trainer.model)
        self.evals = {"loss": 0}
        self.ancestry = [index]

    def advance(self, target):
        self.trainer.config.episodes = target
        self.trainer.run(None)

    def validation(self, regularized=True):
        self.evals["loss"] += 1
        return sum(self.trainer.episode(20000 + i, False, "validation", loss_only=True,
                                        regularized=regularized) for i in range(2)) / 2

    def score(self):
        with torch.no_grad():
            loss = float(self.validation(False).detach())
            bits = float(self.trainer.model.description_cost().detach())
        return loss, bits

    def reseed_from(self, other, sigma):
        model, source = self.trainer.model, other.trainer.model
        model.load_state_dict(copy.deepcopy(source.state_dict()))
        model.temperatures = dict(source.temperatures)
        with torch.no_grad():
            for logits in model.choices:
                logits.add_(torch.randn(logits.shape) * sigma)
        self.trainer.optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.trainer.optimizer.param_groups[0]["lr"])
        original = self.trainer.optimizer.step
        counter = self.steps

        def step(*a, **k):
            counter["n"] += 1
            return original(*a, **k)

        self.trainer.optimizer.step = step
        self.ancestry = other.ancestry + [self.index]


def run(arm, seed, members=4, generations=4, episodes=EPISODES, select="pareto",
        sigma=.5, noise=.5, tolerance=TOLERANCE):
    torch.set_num_threads(1)
    pop = [Member(m, seed, episodes, noise) for m in range(members)]
    generator = torch.Generator().manual_seed(seed * 7919 + 13)
    per = max(1, episodes // max(1, generations))
    log = []
    start = time.perf_counter()

    for gen in range(generations):
        target = min(episodes, per * (gen + 1)) if gen < generations - 1 else episodes
        for member in pop:
            member.advance(target)
        scores = [m.score() for m in pop]
        entry = {"generation": gen, "episodes": target, "scores": scores}
        if select == "pareto" and gen < generations - 1:
            losers = dominated(scores)
            front = [i for i in range(len(pop)) if i not in losers]
            for i in losers:
                parent = front[int(torch.randint(len(front), (1,), generator=generator))]
                pop[i].reseed_from(pop[parent], sigma)
            entry["front"] = front
            entry["reseeded"] = losers
        log.append(entry)

    rows = []
    for member in pop:
        t = member.trainer
        row = {"member": member.index, "ancestry": member.ancestry}
        if arm == "argmax":
            freeze_all_argmax(t.model)
            row["reasons"] = {}
        else:
            scheduler = make_scheduler(arm, t.model, t.optimizer, member.steps, tolerance)
            try:
                scheduler.run(Objective(member.validation, lambda m=member: m.validation(False)),
                              rounds=rounds_for(arm), retrain_steps=RETRAIN_STEPS)
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"scheduler: {exc}"
            row["reasons"] = reason_histogram(scheduler.events)
            row.update(scheduler_record(scheduler))
        n = len(t.model.program.nodes)
        row["frozen"] = len(t.model.frozen)
        row["fully_frozen"] = (len(t.model.frozen) == n
                               and all(not p.requires_grad for p in t.model.constants.values()))
        row["frozen_agent_mean_return"] = None
        if row["fully_frozen"]:
            try:
                mean, returns = frozen_agent_return(t)
                row["frozen_agent_mean_return"] = mean
                row["frozen_agent_returns"] = returns
            except Exception as exc:  # noqa: BLE001
                row["export_error"] = str(exc)
        try:
            row["description_bits"] = t.model.export().pruned().description_bits(t.model.registry)
        except Exception:  # noqa: BLE001
            row["description_bits"] = None
        row["loss"] = float(member.validation(False).detach())
        row["steps"] = member.steps["n"]
        row["forwards"] = member.forwards["n"]
        row["episodes"] = episodes + 2 * member.evals["loss"]
        rows.append(row)

    scored = [r for r in rows if r["frozen_agent_mean_return"] is not None]
    # The population's answer: highest frozen return, ties to the lightest program.
    chosen = (max(scored, key=lambda r: (r["frozen_agent_mean_return"], -(r["description_bits"] or 0)))
              if scored else min(rows, key=lambda r: (r["loss"], r["member"])))
    return {"task": "joint", "arm": f"population/{arm}/{select}", "seed": seed,
            "members": members, "generations": generations, "episodes": episodes,
            "select": select, "sigma": sigma, "init_noise": noise, "member_arm": arm,
            "generation_log": log, "member_rows": rows,
            "chosen_member": chosen["member"],
            "frozen_agent_mean_return": chosen.get("frozen_agent_mean_return"),
            "exact_conformance": chosen.get("frozen_agent_mean_return") is not None,
            "fully_frozen": chosen["fully_frozen"], "frozen": chosen["frozen"],
            "description_bits": chosen["description_bits"],
            "best_member_return": max([r["frozen_agent_mean_return"] for r in scored], default=None),
            "fully_frozen_members": sum(1 for r in rows if r["fully_frozen"]),
            "total_steps": sum(r["steps"] for r in rows),
            "total_forwards": sum(r["forwards"] for r in rows),
            "total_episodes": sum(r["episodes"] for r in rows),
            "wall": time.perf_counter() - start}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="seasons")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--select", default="pareto", choices=("pareto", "none"))
    ap.add_argument("--sigma", type=float, default=.5)
    ap.add_argument("--noise", type=float, default=.5)
    ap.add_argument("--out")
    args = ap.parse_args()
    record = run(args.arm, args.seed, members=args.members, generations=args.generations,
                 episodes=args.episodes, select=args.select, sigma=args.sigma, noise=args.noise)
    text = json.dumps(record)
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
