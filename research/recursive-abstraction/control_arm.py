"""Track 5 control arm C: a SEMANTICALLY USELESS module with the same interface.

Arm B differs from arm A in two ways at once: (i) the search space grows by the
module's call-site candidates, and (ii) those candidates compute the right
sub-function. Arm C isolates (i): it registers a frozen module with the same
typed interface (BOOL,BOOL) -> (BOOL,BOOL) and the same internal size, learned
by the same pipeline, but computing (a and b, a or b) -- useless for the full
adder. Arm C therefore has exactly the same candidate count as arm B.

  A vs C  : cost of merely enlarging the candidate set
  C vs B  : value of the module's semantics
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from tcn.types import BOOL, Value
from tcn.operators import Registry
from tcn.graph import Signal

import experiment as E

DISTRACTOR_SIGNALS = (
    Signal("s", "sum", ("core",), BOOL, "bce"),
    Signal("c", "carry", ("core",), BOOL, "bce"),
)


def distractor_examples():
    rows = []
    for a in (False, True):
        for b in (False, True):
            rows.append(
                {
                    "inputs": {"a": Value.of(BOOL, a), "b": Value.of(BOOL, b)},
                    # deliberately NOT a half adder
                    "targets": {"sum": Value.of(BOOL, a and b), "carry": Value.of(BOOL, a or b)},
                }
            )
    return rows


def build_distractor(seed, steps=200):
    r = Registry()
    prog = E.half_adder_scaffold(r)
    ex = distractor_examples()
    model, opt, rep, exported = E.search(prog, ex, DISTRACTOR_SIGNALS, r, seed, steps=steps, eval_every=5)
    if not rep["final_conformant"]:
        return None, rep
    E.crystallize(model, opt, ex, DISTRACTOR_SIGNALS, r, rounds=6, retrain_steps=5)
    module = model.export()
    if not E.conformant(module, ex, DISTRACTOR_SIGNALS, r):
        return None, rep
    rep["module_digest"] = module.digest
    rep["module_description_bits"] = module.description_bits()
    rep["module_execution_cost"] = module.execution_cost(r)
    return module, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--out", default="control_results.json")
    args = ap.parse_args()
    out = {"config": vars(args), "module_stage": [], "runs": []}
    for seed in range(args.seeds):
        mod, mrep = build_distractor(1000 + seed)
        out["module_stage"].append({k: v for k, v in mrep.items() if k != "curve"})
        if mod is None:
            print(f"[distractor] seed={seed} FAILED", flush=True)
            continue
        t = time.perf_counter()
        rep = E.run_arm("B", seed, args.steps, do_crystallize=False, module=mod)
        rep["arm"] = "C"
        rep["total_wall_seconds"] = time.perf_counter() - t
        out["runs"].append(rep)
        print(f"[C] seed={seed} conf={rep['final_conformant']} first={rep['first_conformant_step']} "
              f"loss={rep['final_loss']:.4f} cands={rep['scaffold_candidates']} "
              f"bits={rep['final_description_bits']} t={rep['total_wall_seconds']:.0f}s", flush=True)
        Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    runs = out["runs"]
    succ = [r for r in runs if r["final_conformant"]]
    print(f"\narm C: success {len(succ)}/{len(runs)}")
    if succ:
        st = [r["first_conformant_step"] for r in succ]
        print(f"  steps median={statistics.median(st)} min={min(st)} max={max(st)}")
    if runs:
        print(f"  final loss median={statistics.median([r['final_loss'] for r in runs]):.4f}")


if __name__ == "__main__":
    main()
