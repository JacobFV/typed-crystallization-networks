"""Ablation on the supervised mixed-synthesis task (`examples/mixed.py`).

This is a local re-implementation of `tcn.synthesis.fit` with identical control
flow and hyper-parameters, plus: seeded choice-logit initialisation (the shipped
fixture is bit-for-bit deterministic, so seeds would otherwise be meaningless),
optimizer-step counting, and an arm switch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.mixed import problem  # noqa: E402
from tcn.learning import SoftProgram, tensor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arms import ARMS, count_steps, freeze_all_argmax, make_scheduler  # noqa: E402

STEPS = 300
LR = 0.05
TOLERANCE = 0.005          # matches tcn.cli.mixed
ROUNDS = 24
RETRAIN_STEPS = 10         # matches tcn.synthesis.fit
ENTROPY_LIMIT = 0.9
INIT_NOISE = 0.5


def build(seed, noise=INIT_NOISE):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    program, signals, examples = problem()
    model = SoftProgram(program)
    if noise:
        with torch.no_grad():
            for logits in model.choices:
                logits.add_(torch.randn(logits.shape) * noise)
    return program, signals, examples, model


def run(arm, seed, steps=STEPS, extra_steps=0, noise=INIT_NOISE):
    program, signals, examples, model = build(seed, noise)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    counter = count_steps(optimizer)

    inputs = {k: torch.stack([tensor(ex["inputs"][k]) for ex in examples]) for k, _ in program.inputs}
    targets = {s.target: torch.stack([tensor(ex["targets"][s.target]) for ex in examples]) for s in signals}

    evals = {"loss": 0, "conform": 0}

    def loss_fn():
        evals["loss"] += 1
        _, _, trace = model(inputs, return_trace=True)
        return model.probe_loss(trace, targets, signals)

    def conform(exact):
        evals["conform"] += 1
        for ex in examples:
            _, _, trace = exact.execute(ex["inputs"], registry=model.registry)
            for signal in signals:
                a = torch.tensor(trace[signal.source].flat())
                b = torch.tensor(ex["targets"][signal.target].flat())
                if float((a - b).abs().max()) > TOLERANCE:
                    return False
        return True

    start = time.perf_counter()
    for step in range(steps):
        optimizer.zero_grad()
        loss = loss_fn() + 0.001 * (step / max(1, steps)) * model.entropy()
        if loss.requires_grad:
            loss.backward()
            optimizer.step()
    base_steps = counter["n"]
    train_wall = time.perf_counter() - start

    record = {
        "task": "mixed", "arm": arm, "seed": seed, "steps": steps,
        "init_noise": noise, "base_steps": base_steps, "extra_steps": extra_steps,
        "soft_loss_before_crystallization": float(loss_fn().detach()),
        "error": None,
    }

    phase = time.perf_counter()
    if arm in ("B", "B0"):
        for _ in range(extra_steps):
            optimizer.zero_grad()
            loss = loss_fn()
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
        try:
            freeze_all_argmax(model)
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"argmax freeze: {exc}"
        record["freeze_events"] = []
        record["trial_steps"] = []
    else:
        scheduler, override = make_scheduler(arm, model, optimizer, counter, seed,
                                             TOLERANCE, ENTROPY_LIMIT)
        rs = RETRAIN_STEPS if override is None else override
        try:
            # Arm H: identical to A except the runtime-conformance acceptance
            # test is not supplied (degradation + connectivity guards intact).
            scheduler.run(loss_fn, rounds=ROUNDS, retrain_steps=rs,
                          conformance=None if arm == "H" else conform)
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"scheduler: {exc}"
        record["freeze_events"] = [asdict(e) for e in scheduler.events]
        record["trial_steps"] = scheduler.trial_steps
        record["retained_trial_steps"] = scheduler.retained_steps()
        record["selection_sweep_evaluations"] = getattr(scheduler, "sweep_evaluations", 0)
    record["crystallization_wall"] = time.perf_counter() - phase
    record["train_wall"] = train_wall
    record["wall"] = time.perf_counter() - start
    record["total_steps"] = counter["n"]
    record["loss_evaluations"] = evals["loss"]
    record["conformance_evaluations"] = evals["conform"]

    n_nodes = len(program.nodes)
    record["nodes"] = n_nodes
    record["frozen"] = len(model.frozen)
    record["frozen_fraction"] = len(model.frozen) / n_nodes
    record["fully_frozen"] = (len(model.frozen) == n_nodes
                              and all(not p.requires_grad for p in model.constants.values()))
    try:
        record["loss"] = float(loss_fn().detach())
    except Exception as exc:  # noqa: BLE001
        record["loss"] = None
        record["loss_error"] = str(exc)
    try:
        record["exact_conformance"] = bool(conform(model.export()))
    except Exception as exc:  # noqa: BLE001
        record["exact_conformance"] = False
        record["export_error"] = str(exc)
    record["selections"] = model.selections()
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true",
                    help="run exactly one (arm, seed) and write a single JSON record")
    ap.add_argument("--arm", default="A")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--arms", default="A,B,B0,C,D,E,F,FR,F2,G,H")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--noise", type=float, default=INIT_NOISE)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "mixed.jsonl"))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.single:
        r = run(args.arm, args.seed, steps=args.steps, extra_steps=args.extra, noise=args.noise)
        out.write_text(json.dumps(r))
        return
    rows = []
    with out.open("w") as fh:
        for seed in range(args.seeds):
            # Arm A first: its total optimizer-step count is the budget every
            # other arm must match (arm B is padded up to it explicitly).
            a = run("A", seed, steps=args.steps, noise=args.noise)
            budget = a["total_steps"]
            fh.write(json.dumps(a) + "\n")
            rows.append(a)
            for arm in args.arms.split(","):
                if arm == "A":
                    continue
                extra = budget - args.steps if arm == "B" else 0
                r = run(arm, seed, steps=args.steps, extra_steps=extra, noise=args.noise)
                r["budget_reference_total_steps"] = budget
                fh.write(json.dumps(r) + "\n")
                rows.append(r)
            fh.flush()
            print(f"seed {seed} done budget={budget}", flush=True)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
