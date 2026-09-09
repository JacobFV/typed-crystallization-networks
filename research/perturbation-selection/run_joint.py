"""Joint prediction+policy fixture (`examples/joint.py`) under each selection rule.

Mirrors the crystallization block of `tcn.cli.joint`: N training episodes, then
crystallization against the mean loss of two held-out validation episodes with
tolerance 0.05, rounds 24, retrain_steps 2. Evaluation is the deterministic mean
return over 16 held-out test episodes plus, when the graph is fully frozen, the
exact frozen-program return over 16 further test episodes.

The `perturbation-total-guard` arm passes the regularized total as a bare
callable, so the connectivity guard probes it instead of the task loss. That is
the shipped pre-change behaviour and isolates the guard's interface change.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from examples.joint import trainer  # noqa: E402
from tcn.agent import Agent  # noqa: E402
from tcn.crystallize import Objective  # noqa: E402
from tcn.generation import Host  # noqa: E402

from arms import (count_forwards, count_steps, freeze_all_argmax,  # noqa: E402
                  make_scheduler, reason_histogram)

EPISODES = 160
TOLERANCE = 0.05
ROUNDS = 24
RETRAIN_STEPS = 2
EVAL_EPISODES = 16


def evaluate(t, n=EVAL_EPISODES):
    rows = [t.episode(10000 + i, train=False, split="test")[0] for i in range(n)]
    return {"mean_return": sum(r["return"] for r in rows) / n,
            "returns": [r["return"] for r in rows],
            "mean_prediction_loss": sum(r["prediction_loss"] for r in rows) / n}


def frozen_agent_return(t, n=EVAL_EPISODES):
    program = t.model.export()
    agent = Agent(program, t.model.registry, t.config)
    returns = []
    for i in range(n):
        host = Host.create(t.config.generator, seed=t.config.seed, index=30000 + i, split="test",
                           configuration=t.config.generator_config | {"horizon": t.config.horizon},
                           objective=t.config.objectives[i % len(t.config.objectives)])
        agent.rollout(host, deterministic=True)
        returns.append(sum(sum(v.decoded for v in r.reward_components.values()) for r in host.records))
    return sum(returns) / n, returns


def run(arm, seed, episodes=EPISODES, extra_steps=0):
    torch.set_num_threads(1)
    t = trainer(episodes, seed=seed)
    steps_counter = count_steps(t.optimizer)
    forward_counter = count_forwards(t.model)

    start = time.perf_counter()
    t.run(None)
    base_steps = steps_counter["n"]
    base_forwards = forward_counter["n"]

    evals = {"loss": 0}

    def validation(regularized=True):
        evals["loss"] += 1
        return sum(t.episode(20000 + i, False, "validation", loss_only=True, regularized=regularized)
                   for i in range(2)) / 2

    record = {"task": "joint", "arm": arm, "seed": seed, "episodes": episodes,
              "base_steps": base_steps, "base_forwards": base_forwards,
              "extra_steps": extra_steps, "pre_crystallization": evaluate(t),
              "final_train_prediction_loss": sum(h["prediction_loss"] for h in t.history[-8:]) / 8,
              "error": None}
    eval_forwards = forward_counter["n"] - base_forwards

    phase = time.perf_counter()
    if arm.startswith("argmax"):
        for _ in range(extra_steps):
            t.optimizer.zero_grad()
            loss = validation()
            if loss.requires_grad:
                loss.backward()
                t.optimizer.step()
        try:
            freeze_all_argmax(t.model)
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"argmax freeze: {exc}"
        record["freeze_events"] = []
        record["sweep_evaluations"] = 0
    else:
        scheduler = make_scheduler(arm, t.model, t.optimizer, steps_counter, TOLERANCE)
        objective = (validation if arm.endswith("total-guard")
                     else Objective(validation, lambda: validation(False)))
        try:
            scheduler.run(objective, rounds=ROUNDS, retrain_steps=RETRAIN_STEPS)
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"scheduler: {exc}"
        record["freeze_events"] = [asdict(e) for e in scheduler.events]
        record["reasons"] = reason_histogram(scheduler.events)
        record["trial_steps"] = scheduler.trial_steps
        record["retained_trial_steps"] = scheduler.retained_steps()
        record["sweep_evaluations"] = scheduler.sweep_evaluations
    record["crystallization_wall"] = time.perf_counter() - phase
    record["crystallization_forwards"] = forward_counter["n"] - base_forwards - eval_forwards

    n = len(t.model.program.nodes)
    record["nodes"] = n
    record["frozen"] = len(t.model.frozen)
    record["frozen_fraction"] = len(t.model.frozen) / n
    record["fully_frozen"] = (len(t.model.frozen) == n
                              and all(not p.requires_grad for p in t.model.constants.values()))
    try:
        record["post_crystallization"] = evaluate(t)
    except Exception as exc:  # noqa: BLE001
        record["post_crystallization"] = None
        record["eval_error"] = str(exc)
    record["exact_conformance"] = False
    record["frozen_agent_mean_return"] = None
    if record["fully_frozen"]:
        try:
            mean, returns = frozen_agent_return(t)
            record["frozen_agent_mean_return"] = mean
            record["frozen_agent_returns"] = returns
            record["exact_conformance"] = True
        except Exception as exc:  # noqa: BLE001
            record["export_error"] = str(exc)
    record["wall"] = time.perf_counter() - start
    record["total_steps"] = steps_counter["n"]
    record["total_forwards"] = forward_counter["n"]
    record["loss_evaluations"] = evals["loss"]
    record["selections"] = t.model.selections()
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="perturbation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--out")
    args = ap.parse_args()
    record = run(args.arm, args.seed, episodes=args.episodes, extra_steps=args.extra)
    text = json.dumps(record)
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
