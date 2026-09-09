"""Why does `runtime conformance` reject freeze trials?

`tcn/synthesis.py:fit` passes a conformance callback that runs
`model.export()` -- which hardens EVERY node by argmax, frozen or not -- and
requires every declared signal to match its target exactly on every training
example. This script records, for each rejected trial, which signal disagreed
and whether that signal's node was already frozen, on trial, or still soft.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from arms import Instrumented, count_steps  # noqa: E402
from run_mixed import ENTROPY_LIMIT, LR, RETRAIN_STEPS, ROUNDS, TOLERANCE, build  # noqa: E402
from tcn.learning import tensor  # noqa: E402


def diagnose(seed, steps):
    program, signals, examples, model = build(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    counter = count_steps(optimizer)
    inputs = {k: torch.stack([tensor(ex["inputs"][k]) for ex in examples]) for k, _ in program.inputs}
    targets = {s.target: torch.stack([tensor(ex["targets"][s.target]) for ex in examples]) for s in signals}

    def loss_fn():
        _, _, trace = model(inputs, return_trace=True)
        return model.probe_loss(trace, targets, signals)

    for step in range(steps):
        optimizer.zero_grad()
        loss = loss_fn() + 0.001 * (step / max(1, steps)) * model.entropy()
        if loss.requires_grad:
            loss.backward()
            optimizer.step()

    failures = []
    state = {"current": None}

    class Tracked(Instrumented):
        def try_freeze(self, name, loss_fn, retrain_steps=20, conformance=None):
            state["current"] = name
            try:
                return super().try_freeze(name, loss_fn, retrain_steps, conformance)
            finally:
                state["current"] = None

    def conform(exact):
        worst = {}
        for ex in examples:
            _, _, trace = exact.execute(ex["inputs"], registry=model.registry)
            for s in signals:
                a = torch.tensor(trace[s.source].flat())
                b = torch.tensor(ex["targets"][s.target].flat())
                err = float((a - b).abs().max())
                if err > worst.get(s.source, (-1, None))[0]:
                    worst[s.source] = (err, {k: v.decoded for k, v in ex["inputs"].items()},
                                       a.tolist(), b.tolist())
        bad = {k: v for k, v in worst.items() if v[0] > TOLERANCE}
        if bad:
            trial = state["current"]
            committed = sorted(n for n in model.frozen if n != trial)
            failures.append({
                "node_under_trial": trial,
                "previously_frozen": committed,
                "mismatched_signals": sorted(bad),
                "mismatch_on_trial_node": trial in bad,
                "mismatch_on_previously_frozen": sorted(k for k in bad if k in committed),
                "mismatch_only_on_untouched_soft_nodes": sorted(
                    k for k in bad if k != trial and k not in committed),
                "worst": {k: {"abs_error": v[0], "inputs": v[1], "exact": v[2], "target": v[3]}
                          for k, v in bad.items()},
            })
            return False
        return True

    scheduler = Tracked(model, optimizer, tolerance=TOLERANCE,
                             entropy_limit=ENTROPY_LIMIT, counter=counter)
    scheduler.run(loss_fn, rounds=ROUNDS, retrain_steps=RETRAIN_STEPS, conformance=conform)
    return failures, scheduler, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "results" / "conformance-diagnosis.json"))
    args = ap.parse_args()

    summary = {"steps": args.steps, "seeds": args.seeds, "runs": []}
    roles = Counter()
    for seed in range(args.seeds):
        failures, scheduler, model = diagnose(seed, args.steps)
        rejected = [e for e in scheduler.events if not e.accepted]
        for f in failures:
            if f["mismatch_on_previously_frozen"]:
                roles["mismatch on an earlier committed freeze"] += 1
            elif f["mismatch_on_trial_node"] and f["mismatch_only_on_untouched_soft_nodes"]:
                roles["mismatch on the trial node AND on untouched soft nodes"] += 1
            elif f["mismatch_on_trial_node"]:
                roles["mismatch on the node under trial only (legitimate rejection)"] += 1
            else:
                roles["mismatch ONLY on untouched still-soft nodes (checker artifact)"] += 1
        summary["runs"].append({
            "seed": seed,
            "trials": len(scheduler.events),
            "rejected": len(rejected),
            "conformance_rejections": sum(1 for e in rejected if e.reason == "runtime conformance"),
            "frozen_at_end": sorted(model.frozen),
            "first_failures": failures[:3],
        })
    summary["mismatch_role_histogram"] = dict(roles)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps({"mismatch_role_histogram": dict(roles),
                      "runs": [{k: v for k, v in r.items() if k != "first_failures"}
                               for r in summary["runs"]]}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
