"""What the selector chooses, what was right, and what being wrong costs.

Every case is run through **both** backends wherever both can be afforded, so
the cost of the selector's mistakes is measured rather than argued. Where one
backend is priced out, its projection is recorded and labelled as a projection.

Scoring, stated up front so it cannot be adjusted afterwards:

  * a backend **succeeds** on a case when the program it returns is exact on the
    validation split it never fitted (or on training, where the case has no
    held-out data). Relaxed loss is never used -- FINDINGS section 4 records
    17/17 failures reaching zero soft BCE with a wrong argmax.
  * the selector is **right** on a case when it picks a backend that succeeds and
    the alternative is not strictly cheaper at equal success.
  * the **cost of being wrong** is measured in the currency the case is scarce
    in: seconds for CPU-bound cases, environment steps for the coupled one.

MEASUREMENT WARNING. `SoftProgram` zero-initializes every choice logit, so
`torch.manual_seed` does not vary synthesis. Every gradient arm here adds
explicit initialization noise of 0.5 through `common.perturb` and reports a
rate over SEEDS runs; the discrete arms are deterministic and report one
outcome.

Run:  PYTHONPATH=. .venv/bin/python research/search-selection/suite.py
"""
from __future__ import annotations
import math
import time

import torch

from common import CASES, dump, exact_error, perturb, selected_error
from tcn.learning import SoftProgram, tensor
from tcn.search import enumerate_fit, space_size
from tcn.select import hybrid_fit, select_backend

SEEDS = 3
STEPS = 400
LR = .05
ENUMERATION_CAP = 300.          # seconds we are willing to spend measuring an arm


def relax_arm(case, seeds=SEEDS, steps=None, lr=LR, carrier_scaled=False):
    """Plain gradient search over the same candidate space, then argmax.

    Deliberately not `fit(freeze=True)`: crystallization is a separate mechanism
    that FINDINGS section 3 measures as contributing nothing at working budgets,
    and including it would price the scheduler rather than the relaxation. This
    is the arm the perception-ladder and address-wall tracks ran.
    """
    program, signals, r = case.program, case.signals, case.registry
    steps = case.relax_steps if steps is None else steps
    inputs = {k: torch.stack([tensor(e["inputs"][k]) for e in case.train]) for k, _ in program.inputs}
    targets = {s.target: torch.stack([tensor(e["targets"][s.target]) for e in case.train]) for s in signals}
    held = case.held_out
    rows = []
    t0 = time.perf_counter()
    for seed in range(seeds):
        m = perturb(SoftProgram(program, r), seed)
        if carrier_scaled: m.scale_surrogates()
        params = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
        started = time.perf_counter()
        failed = None
        for _ in range(steps):
            opt.zero_grad()
            try:
                _, _, tr = m(inputs, return_trace=True)
                loss = m.probe_loss(tr, targets, signals)
            except (ValueError, RuntimeError) as e:
                failed = str(e)[:80]; break
            if not loss.requires_grad: break
            loss.backward(); opt.step()
        sel = m.selections()
        err = float("inf") if failed else exact_error(m.export(), held, signals, r)
        rows.append({"seed": seed, "selections": sel, "held_out_max_error": err,
                     "exact": err <= case.tolerance, "seconds": round(time.perf_counter() - started, 3),
                     "failure": failed})
    return {"backend": "relax", "carrier_scaled": carrier_scaled, "seeds": seeds, "steps": steps,
            "successes": sum(x["exact"] for x in rows), "seconds": round(time.perf_counter() - t0, 3),
            "seconds_per_run": round((time.perf_counter() - t0) / max(1, seeds), 3),
            "certificate": False, "rows": rows}


def enumerate_arm(case, decision, cap=ENUMERATION_CAP):
    if decision.projected_enumeration_seconds > cap:
        return {"backend": "enumerate", "ran": False,
                "projected_seconds": decision.projected_enumeration_seconds,
                "seconds_per_program": decision.seconds_per_program,
                "note": f"priced out at {decision.projected_enumeration_seconds:.4g}s against a "
                        f"{cap}s measurement cap; the projection is reported instead"}
    scored = list(case.train) + list(case.validation)
    t0 = time.perf_counter()
    res = enumerate_fit(case.program, scored, case.signals, case.registry, case.tolerance)
    held = case.held_out
    err = (selected_error(case.program, res.selections, held, case.signals, case.registry)
           if res.solved else float("inf"))
    # what enumeration would have returned had it been scored on training alone
    t1 = time.perf_counter()
    train_only = enumerate_fit(case.program, case.train, case.signals, case.registry, case.tolerance)
    train_only_err = (selected_error(case.program, train_only.selections, held, case.signals, case.registry)
                      if train_only.solved else float("inf"))
    return {"backend": "enumerate", "ran": True, "seconds": round(time.perf_counter() - t0 - (time.perf_counter() - t1), 4),
            "successes": int(res.solved and err <= case.tolerance), "seeds": 1,
            "held_out_max_error": err, "solved": res.solved, "unique": res.unique,
            "conforming": res.conforming, "exhausted": res.exhausted,
            "certificate": bool(res.exhausted and res.unique),
            "train_only": {"seconds": round(time.perf_counter() - t1, 4), "solved": train_only.solved,
                           "conforming": train_only.conforming, "unique": train_only.unique,
                           "held_out_max_error": train_only_err,
                           "exact": train_only_err <= case.tolerance},
            "selections": res.selections}


def hybrid_arm(case, constant_steps=200):
    t0 = time.perf_counter()
    res, constants = hybrid_fit(case.program, list(case.train) + list(case.validation), case.signals,
                                case.registry, case.tolerance, constant_steps=constant_steps)
    err = float("inf")
    if res.solved:
        m = SoftProgram(case.program, case.registry)
        for k, v in res.selections.items(): m.freeze(k, v)
        with torch.no_grad():
            for k, v in constants.items(): m.constants[k].copy_(v)
        err = exact_error(m.export(), case.held_out, case.signals, case.registry)
    return {"backend": "hybrid", "ran": True, "seconds": round(time.perf_counter() - t0, 3),
            "successes": int(res.solved and err <= case.tolerance), "seeds": 1,
            "held_out_max_error": err, "solved": res.solved, "evaluated": res.evaluated,
            "certificate": False, "constant_steps": constant_steps,
            "constants": {k: [round(float(x), 8) for x in v.flatten().tolist()] for k, v in (constants or {}).items()},
            "selections": res.selections}


def main():
    t0 = time.perf_counter()
    out = {"seeds": SEEDS, "steps": STEPS, "cases": []}
    for name, make in CASES.items():
        case = make()
        t = time.perf_counter()
        decision = select_backend(case.program, case.train, case.signals, case.registry,
                                  case.tolerance, rollout_cost=case.rollout_cost)
        decide_seconds = time.perf_counter() - t
        block = {"case": name, "family": case.family, "space": case.space,
                 "train": len(case.train), "validation": len(case.validation),
                 "test": len(case.test), "held_out_is_independent": case.held_out_is_independent,
                 "relax_steps": case.relax_steps,
                 "right": case.right, "evidence": case.evidence,
                 "rollout_cost": case.rollout_cost,
                 "chosen": decision.mode, "reason": decision.reason,
                 "decision_seconds": round(decide_seconds, 3),
                 "decision": {k: v for k, v in decision.to_dict().items() if k != "liveness"},
                 "arms": []}
        print(f"== {name}  space={case.space}  chose {decision.mode} "
              f"(right: {case.right}) in {decide_seconds:.2f}s", flush=True)
        print("   ", decision.reason, flush=True)

        block["arms"].append(enumerate_arm(case, decision))
        if case.program.trainable_constants and case.space <= 4096:
            block["arms"].append(hybrid_arm(case))
        block["arms"].append(relax_arm(case, carrier_scaled=decision.scale_surrogates))
        if decision.scale_surrogates:
            block["arms"].append(relax_arm(case, carrier_scaled=False) | {"backend": "relax_shipped_surrogate"})
        for a in block["arms"]:
            print(f"    {a['backend']:26s} ran={a.get('ran', True)} "
                  f"successes={a.get('successes')}/{a.get('seeds')} "
                  f"seconds={a.get('seconds', a.get('projected_seconds'))} "
                  f"certificate={a.get('certificate')}", flush=True)
        out["cases"].append(block)
    out["wall_seconds"] = round(time.perf_counter() - t0, 2)
    dump("suite", out)


if __name__ == "__main__":
    main()
