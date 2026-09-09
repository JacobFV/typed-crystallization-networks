"""Mixed-synthesis fixture (`examples/mixed.py`) under each commitment schedule.

A local re-implementation of `tcn.synthesis.fit` with the same control flow and
hyper-parameters, plus optimizer-step / forward-pass counting, seeded
choice-logit initialisation and the arm switch. The fixture has no trainable
constants, so `fit`'s constant-polish phase is a no-op here and is omitted.

The shipped fixture is bit-for-bit deterministic -- zero-initialised logits,
full-batch training, no sampling anywhere -- so seeds mean nothing without added
variation. Every run therefore adds seeded Gaussian noise (std 0.5, track 1's
value) to the choice logits at initialisation and changes nothing else; the
`--noise 0` control reproduces the shipped single deterministic outcome.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from examples.mixed import problem  # noqa: E402
from tcn.crystallize import Objective  # noqa: E402
from tcn.learning import SoftProgram, tensor  # noqa: E402

from arms import (count_forwards, count_steps, freeze_all_argmax,  # noqa: E402
                  make_scheduler, reason_histogram, rounds_for)

STEPS = 300
LR = 0.05
TOLERANCE = 0.005      # tcn.cli.mixed
RETRAIN_STEPS = 10     # tcn.synthesis.fit
INIT_NOISE = 0.5


def run(arm, seed, steps=STEPS, extra_steps=0, noise=INIT_NOISE):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    program, signals, examples = problem()
    model = SoftProgram(program)
    if noise:
        with torch.no_grad():
            for logits in model.choices:
                logits.add_(torch.randn(logits.shape) * noise)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    steps_counter = count_steps(optimizer)
    forward_counter = count_forwards(model)

    inputs = {k: torch.stack([tensor(ex["inputs"][k]) for ex in examples]) for k, _ in program.inputs}
    targets = {s.target: torch.stack([tensor(ex["targets"][s.target]) for ex in examples]) for s in signals}
    evals = {"loss": 0, "conform": 0}

    def loss_fn():
        evals["loss"] += 1
        _, _, trace = model(inputs, return_trace=True)
        return model.probe_loss(trace, targets, signals)

    def exact_error(exact):
        evals["conform"] += 1
        worst = 0.0
        for ex in examples:
            _, _, trace = exact.execute(ex["inputs"], registry=model.registry)
            for signal in signals:
                a = torch.tensor(trace[signal.source].flat())
                b = torch.tensor(ex["targets"][signal.target].flat())
                worst = max(worst, float((a - b).abs().max()))
        return worst

    start = time.perf_counter()
    for step in range(steps):
        optimizer.zero_grad()
        loss = loss_fn() + 0.001 * (step / max(1, steps)) * model.entropy()
        if loss.requires_grad:
            loss.backward()
            optimizer.step()
    base_steps = steps_counter["n"]
    base_forwards = forward_counter["n"]

    record = {"task": "mixed", "arm": arm, "seed": seed, "steps": steps, "init_noise": noise,
              "base_steps": base_steps, "base_forwards": base_forwards,
              "extra_steps": extra_steps, "error": None}

    phase = time.perf_counter()
    if arm.startswith("argmax"):
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
        record["sweep_evaluations"] = 0
        record["gate_evaluations"] = 0
    else:
        scheduler = make_scheduler(arm, model, optimizer, steps_counter, TOLERANCE)
        # The supervised objective carries no architecture regularizer (the
        # entropy term is added in the training loop, not during retraining), so
        # total and task are the same closure -- as `tcn.synthesis.fit` states.
        try:
            scheduler.run(Objective(loss_fn, loss_fn), rounds=rounds_for(arm),
                          retrain_steps=RETRAIN_STEPS,
                          conformance=lambda exact: exact_error(exact) <= TOLERANCE)
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"scheduler: {exc}"
        record["freeze_events"] = [asdict(e) for e in scheduler.events]
        record["reasons"] = reason_histogram(scheduler.events)
        record["trial_steps"] = scheduler.trial_steps
        record["retained_trial_steps"] = scheduler.retained_steps()
        record["sweep_evaluations"] = scheduler.sweep_evaluations
        record["gate_evaluations"] = scheduler.gate_evaluations
        record["gate_log"] = scheduler.gate_log
        record["rounds"] = len(scheduler.gate_log)
        record["open_rounds"] = sum(1 for g in scheduler.gate_log if g["open"])
    record["crystallization_wall"] = time.perf_counter() - phase
    record["wall"] = time.perf_counter() - start

    n = len(program.nodes)
    record["nodes"] = n
    record["frozen"] = len(model.frozen)
    record["frozen_fraction"] = len(model.frozen) / n
    record["fully_frozen"] = (len(model.frozen) == n
                              and all(not p.requires_grad for p in model.constants.values()))
    try:
        record["loss"] = float(loss_fn().detach())
    except Exception as exc:  # noqa: BLE001
        record["loss"] = None
        record["loss_error"] = str(exc)
    try:
        err = exact_error(model.export())
        record["exact_max_error"] = err
        record["exact_conformance"] = err <= TOLERANCE
    except Exception as exc:  # noqa: BLE001
        record["exact_max_error"] = None
        record["exact_conformance"] = False
        record["export_error"] = str(exc)
    record["total_steps"] = steps_counter["n"]
    record["total_forwards"] = forward_counter["n"]
    record["crystallization_forwards"] = forward_counter["n"] - base_forwards
    record["loss_evaluations"] = evals["loss"]
    record["conformance_evaluations"] = evals["conform"]
    record["selections"] = model.selections()
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="perturbation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--noise", type=float, default=INIT_NOISE)
    ap.add_argument("--out")
    args = ap.parse_args()
    record = run(args.arm, args.seed, steps=args.steps, extra_steps=args.extra, noise=args.noise)
    text = json.dumps(record)
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
