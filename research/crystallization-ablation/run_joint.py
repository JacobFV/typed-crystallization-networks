"""Ablation on the joint prediction+policy logic task (`examples/joint.py`).

Mirrors `tcn.cli.joint`: 160 training episodes, then crystallization against the
mean loss of two held-out validation episodes with tolerance=0.05,
entropy_limit=0.9, rounds=24, retrain_steps=2. Evaluation is the deterministic
mean return over 16 held-out test episodes, plus (when the graph is fully
frozen) the exact frozen-agent return over 16 further test episodes.
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

import torch  # noqa: E402

from examples.joint import trainer  # noqa: E402
from tcn.agent import Agent  # noqa: E402
from tcn.generation import Host  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arms import count_steps, freeze_all_argmax, make_scheduler  # noqa: E402

EPISODES = 160
TOLERANCE = 0.05
ENTROPY_LIMIT = 0.9
ROUNDS = 24
RETRAIN_STEPS = 2
EVAL_EPISODES = 16


def evaluate(t, n=EVAL_EPISODES):
    rows = [t.episode(10000 + i, train=False, split="test")[0] for i in range(n)]
    return {
        "mean_return": sum(r["return"] for r in rows) / n,
        "returns": [r["return"] for r in rows],
        "mean_prediction_loss": sum(r["prediction_loss"] for r in rows) / n,
    }


def frozen_agent_return(t, n=EVAL_EPISODES):
    program = t.model.export()
    agent = Agent(program, t.model.registry, t.config)
    returns = []
    for i in range(n):
        host = Host.create(
            t.config.generator, seed=t.config.seed, index=30000 + i, split="test",
            configuration=t.config.generator_config | {"horizon": t.config.horizon},
            objective=t.config.objectives[i % len(t.config.objectives)],
        )
        agent.rollout(host, deterministic=True)
        returns.append(sum(sum(v.decoded for v in r.reward_components.values()) for r in host.records))
    return sum(returns) / n, returns


def run(arm, seed, episodes=EPISODES, extra_steps=0):
    torch.set_num_threads(1)
    t = trainer(episodes, seed=seed)
    counter = count_steps(t.optimizer)

    start = time.perf_counter()
    t.run(None)
    base_steps = counter["n"]
    train_wall = time.perf_counter() - start

    evals = {"loss": 0}

    def val_loss():
        evals["loss"] += 1
        return sum(t.episode(20000 + i, False, "validation", loss_only=True) for i in range(2)) / 2

    pre = evaluate(t)
    record = {
        "task": "joint", "arm": arm, "seed": seed, "episodes": episodes,
        "base_steps": base_steps, "extra_steps": extra_steps,
        "pre_crystallization": pre,
        "final_train_prediction_loss": sum(h["prediction_loss"] for h in t.history[-8:]) / 8,
        "error": None,
    }

    phase = time.perf_counter()
    if arm in ("B", "B0"):
        for _ in range(extra_steps):
            t.optimizer.zero_grad()
            loss = val_loss()
            if loss.requires_grad:
                loss.backward()
                t.optimizer.step()
        try:
            freeze_all_argmax(t.model)
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"argmax freeze: {exc}"
        record["freeze_events"] = []
        record["trial_steps"] = []
    else:
        scheduler, override = make_scheduler(arm, t.model, t.optimizer, counter, seed,
                                             TOLERANCE, ENTROPY_LIMIT)
        rs = RETRAIN_STEPS if override is None else override
        try:
            scheduler.run(val_loss, rounds=ROUNDS, retrain_steps=rs)
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"scheduler: {exc}"
        record["freeze_events"] = [asdict(e) for e in scheduler.events]
        record["trial_steps"] = scheduler.trial_steps
        record["retained_trial_steps"] = scheduler.retained_steps()
        record["selection_sweep_evaluations"] = getattr(scheduler, "sweep_evaluations", 0)
    record["crystallization_wall"] = time.perf_counter() - phase

    n_nodes = len(t.model.program.nodes)
    record["nodes"] = n_nodes
    record["frozen"] = len(t.model.frozen)
    record["frozen_fraction"] = len(t.model.frozen) / n_nodes
    record["fully_frozen"] = (len(t.model.frozen) == n_nodes
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
    record["train_wall"] = train_wall
    record["wall"] = time.perf_counter() - start
    record["total_steps"] = counter["n"]
    record["loss_evaluations"] = evals["loss"]
    record["selections"] = t.model.selections()
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true",
                    help="run exactly one (arm, seed) and write a single JSON record")
    ap.add_argument("--arm", default="A")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--arms", default="A,B,B0,C,D,E,F,FR,F2,G")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "joint.jsonl"))
    args = ap.parse_args()

    arms = args.arms.split(",")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.single:
        r = run(args.arm, args.seed, episodes=args.episodes, extra_steps=args.extra)
        out.write_text(json.dumps(r))
        return
    with out.open("w") as fh:
        for seed in range(args.seeds):
            a = run("A", seed, episodes=args.episodes)
            budget = a["total_steps"]
            fh.write(json.dumps(a) + "\n")
            for arm in arms:
                if arm == "A":
                    continue
                extra = budget - args.episodes if arm == "B" else 0
                r = run(arm, seed, episodes=args.episodes, extra_steps=extra)
                r["budget_reference_total_steps"] = budget
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"seed {seed} done budget={budget} A_return={a['post_crystallization']['mean_return']}", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
