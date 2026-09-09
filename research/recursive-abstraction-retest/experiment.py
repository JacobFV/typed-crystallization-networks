"""Track 5 retest: matched arms on a verified non-collapsing composite.

Target   G(a,b,c,d,e,f) = MAJ3(a,b,c) xor MAJ3(d,e,f)
         verified flat minimum in [7, 9] nodes (min_program.py: gate-elimination
         lower bound 7 over the full binary basis, explicit 9-gate circuit
         executed against all 64 rows). Disjoint argument windows, so it does
         not collapse the way track 5's overlapping-window composite did.

Module   MAJ3, verified minimum 4 gates, not available as a primitive.
Route    2 module calls + 1 xor = 3 live nodes against 7-9 flat nodes.

Arms, sharing one scaffold with identical nodes, depths and pools:
  A  no module candidates -- the flat route only
  B  MAJ3 offered as a BOOL-valued candidate at every node, over the six inputs
  C  the same, with a verified-useless module of the same size (truth table 134,
     minimum 4 gates, and shortens MAJ3 by nothing)

Arms B and C are supplied with the *minimum* circuit for their sub-function,
verified by `min_program.py` and re-checked against the truth table at
construction. Acquisition is measured separately (`--acquire`) rather than
folded into the arms, for two reasons that are themselves results: the learned
MAJ3 keeps a dead gate even after pruning (5 live nodes where 4 suffice), which
would charge abstraction for track 5's F4; and arm C's sub-function is not
reliably learnable by this synthesizer at all (0/8 attempts, 6400 steps), so a
learned arm C could not be matched to arm B.

Scaffolds:
  wide   8 workhorse nodes + an output node; both routes fit
  tight  3 nodes; below the verified flat minimum, so ONLY the abstracted route
         can solve it -- and small enough for exhaustive enumeration
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from tcn.operators import Registry
from tcn.graph import Program

import common as C

HERE = Path(__file__).parent
FN = {"B": C.maj, "C": C.distractor}


# --------------------------------------------------------------------------
def acquire(job):
    """Measure what it costs to acquire the sub-module, per seed, per arm."""
    arm, seed = job
    torch.set_num_threads(1)
    t = time.perf_counter()
    module, rep = C.acquire_module(FN[arm], seed)
    rep["arm"] = arm
    rep["wall_seconds"] = time.perf_counter() - t
    rep["acquired"] = module is not None
    if module is not None:
        r = Registry()
        rep["semantics_verified"] = C.verify_module(module, FN[arm], r)
        rep["minimum_nodes"] = 4
    return arm, seed, rep


def run_one(job):
    arm, seed, scaffold, steps = job
    torch.set_num_threads(1)
    r = Registry()
    name = None
    if arm in FN:
        module = C.minimal_module(r, "maj" if arm == "B" else "distractor")
        name = r.register_module(module)
        assert C.verify_module(module, FN[arm], r)
    build = C.wide_scaffold if scaffold == "wide" else C.tight_scaffold
    prog = build(r, name)
    ex = C.composite_examples()
    t = time.perf_counter()
    model, opt, rep, exported = C.search(prog, ex, C.COMP_SIGNALS, r, seed,
                                         steps=steps, eval_every=10, trace_every=10)
    pruned = C.prune(exported)
    sel = model.selections()
    rep.update(
        arm=arm, seed=seed, scaffold=scaffold,
        scaffold_nodes=len(prog.nodes),
        scaffold_candidates=sum(len(n.candidates) for n in prog.nodes),
        module_selected_somewhere=any(
            n.candidates[sel[n.name]].operator.name.startswith("module:") for n in prog.nodes),
        module_on_output_path=C.module_on_output_path(exported),
        live_nodes=len(pruned.nodes),
        description_bits_pruned=pruned.description_bits(r),
        description_bits_pruned_no_library=pruned.description_bits(),
        execution_cost_pruned=pruned.execution_cost(r),
        total_wall_seconds=time.perf_counter() - t,
        selections={n.name: {"operator": n.candidates[sel[n.name]].operator.name[:20],
                             "sources": list(n.candidates[sel[n.name]].sources)}
                    for n in prog.nodes},
        live_selections={n.name: {"operator": n.candidates[n.selected or 0].operator.name[:20],
                                  "sources": list(n.candidates[n.selected or 0].sources)}
                         for n in pruned.nodes},
    )
    return rep


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--tight-steps", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--acquire", action="store_true", help="measure module acquisition cost too")
    ap.add_argument("--scaffolds", default="wide,tight")
    ap.add_argument("--arms", default="A,B,C")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    out = {"config": vars(args), "modules": [], "runs": []}

    # ---- module acquisition, measured but not folded into the arms ------
    if args.acquire:
        jobs = [(arm, seed) for arm in ("B", "C") for seed in range(args.seeds)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for arm, seed, rep in pool.map(acquire, jobs):
                out["modules"].append({k: v for k, v in rep.items() if k != "crystallization"})
                print(f"[acquire {arm}] seed={seed} ok={rep['acquired']} "
                      f"attempts={rep['attempts']} steps={rep['acquisition_steps']} "
                      f"nodes={rep.get('pruned_nodes')} bits={rep.get('description_bits')} "
                      f"cost={rep.get('execution_cost')} t={rep['wall_seconds']:.0f}s", flush=True)
        Path(args.out).write_text(json.dumps(out, indent=2, default=str))

    # ---- the matched arms ------------------------------------------------
    runs = []
    for scaffold in args.scaffolds.split(","):
        steps = args.steps if scaffold == "wide" else args.tight_steps
        for seed in range(args.seeds):
            for arm in args.arms.split(","):
                runs.append((arm, seed, scaffold, steps))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rep in pool.map(run_one, runs):
            out["runs"].append(rep)
            print(f"[{rep['scaffold']} {rep['arm']}] seed={rep['seed']} "
                  f"conf={rep['final_conformant']} first={rep['first_conformant_step']} "
                  f"loss={rep['final_loss']:.4f} live={rep['live_nodes']} "
                  f"mod_sel={rep['module_selected_somewhere']} "
                  f"mod_path={rep['module_on_output_path']} "
                  f"t={rep['total_wall_seconds']:.0f}s", flush=True)
            Path(args.out).write_text(json.dumps(out, indent=2, default=str))

    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    summarize(out)


def summarize(out):
    for scaffold in ("wide", "tight"):
        print(f"\n=== {scaffold} scaffold ===")
        for arm in ("A", "B", "C"):
            rs = [r for r in out["runs"] if r["arm"] == arm and r["scaffold"] == scaffold]
            if not rs:
                continue
            succ = [r for r in rs if r["final_conformant"]]
            print(f"arm {arm}: success {len(succ)}/{len(rs)}  "
                  f"module_on_output_path {sum(r['module_on_output_path'] for r in rs)}/{len(rs)}  "
                  f"(of successes {sum(r['module_on_output_path'] for r in succ)}/{len(succ)})")
            print(f"   loss median={statistics.median([r['final_loss'] for r in rs]):.4f}  "
                  f"sec/step={statistics.median([r['seconds_per_step'] for r in rs]):.3f}  "
                  f"cands={rs[0]['scaffold_candidates']}")
            if succ:
                st = [r["first_conformant_step"] for r in succ]
                print(f"   steps to conformance: median={statistics.median(st)} range={min(st)}-{max(st)}")


if __name__ == "__main__":
    main()
