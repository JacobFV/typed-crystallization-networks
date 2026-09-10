"""The decisive measurement: does each objective's rank-1 class help the task
that was removed from the corpus it was mined from?

§54's protocol, reused rather than rebuilt: the two-node compact scaffold,
`tcn.search.enumerate_fit` with `tolerance=1e-3`, `rank="order"` and
`max_programs` far above the space size so every run is a **full sweep** and
the certificate survives.  A module selected on the corpus `band \\ t` is
offered to `t`, which contributed no program at any length to that corpus.

Beside every objective: the floor (`none`, the flat compact scaffold), the
ceiling (the hand-authored `W4`), the hand-authored wrong module `X4`, and the
two off-family controls `H_par5` and `H_x4`, which every `W4`-family arm must
leave at 0.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _paths  # noqa: F401

from tcn.library import Library
from tcn.operators import Registry
from tcn.search import candidate_counts, enumerate_fit, space_size

import later

import armlib
import evaltasks

OUT = Path(__file__).resolve().parent / "out"

OBJECTIVES = ("O1", "O2", "O3", "O4", "B1", "B2")
CONTROLS = ("H_par5", "H_x4")


def registry_for(band, spec):
    r = Registry()
    if spec == "none":
        return r, None
    kind, name = spec.split(":", 1)
    if kind == "hand":
        return r, r.register_module(armlib.hand_authored(r, name))
    _, aliases = Library(armlib.lib_for(band)).load([name], registry=r,
                                                    policy="strict")
    return r, aliases[name]


def _enum(job):
    band, task, spec = job
    fn = evaltasks.COMPACT_TASKS[task]
    r, module_name = registry_for(band, spec)
    program = evaltasks.compact_scaffold(r, module_name)
    t0 = time.perf_counter()
    res = enumerate_fit(program, evaltasks.examples_for(fn), evaltasks.SIGNALS,
                        registry=r, tolerance=.001, max_programs=1 << 24,
                        rank="order")
    row = res.to_dict()
    row.update(band=band, task=task, spec=spec, module=module_name,
               candidates_per_node=list(candidate_counts(program)),
               declared_space_size=space_size(program),
               constant_baseline=evaltasks.constant_baseline(fn),
               random_baseline=0.5, wall_seconds=time.perf_counter() - t0)
    if res.selections:
        row["module_on_output_path"] = later.module_on_output_path(
            program.harden(res.selections))
    return row


def _grad(job):
    band, task, spec, seed, steps = job
    fn = evaltasks.COMPACT_TASKS[task]
    r, module_name = registry_for(band, spec)
    program = evaltasks.compact_scaffold(r, module_name)
    rec = later.gradient_search(program, evaltasks.examples_for(fn),
                                evaltasks.SIGNALS, r, seed, steps=steps)
    rec.update(band=band, task=task, spec=spec, space_size=space_size(program))
    return rec


def plan_for(band):
    ranked = json.loads((OUT / f"ranked_{band}.json").read_text())
    plan, specs = [], set()
    for t in evaltasks.LOO_TASKS:
        cname = "wo_" + t.split("_")[0]
        for oid in OBJECTIVES:
            tab = ranked["corpora"][cname]["tables"][oid]
            top = tab["rank1"]
            spec = "lib:" + top["library_label"]
            plan.append({"band": band, "objective": oid, "corpus": cname,
                         "held_out": t, "spec": spec, "digest": top["digest"],
                         "arity": top["arity"], "nodes": top["nodes"],
                         "tasks_in_class": len(top["tasks"]),
                         "is_window": top["is_window"],
                         "best_window_rank": tab["best_window_rank"],
                         "rank1_decided_by": tab["rank1_decided_by"]})
            specs.add(spec)
        # EXPLORATORY, as §52 ran it: the best *window* class in the same
        # leave-one-out table whatever its rank -- separates "the corpus does
        # not contain it" from "the ranking does not pick it".
        rows = ranked["corpora"][cname]["tables"]["O1"]["ranked"]
        win = [r for r in rows if r["is_window"]]
        if win:
            spec = "lib:cls_" + win[0]["digest"][:12]
            plan.append({"band": band, "objective": "WIN", "corpus": cname,
                         "held_out": t, "spec": spec, "digest": win[0]["digest"],
                         "arity": win[0]["arity"], "nodes": win[0]["nodes"],
                         "tasks_in_class": len(win[0]["tasks"]),
                         "is_window": True, "best_window_rank": win[0]["rank"],
                         "exploratory": True})
            specs.add(spec)
    for oid in OBJECTIVES:
        specs.add("lib:" + ranked["corpora"]["full"]["tables"][oid]["rank1"]["library_label"])
    specs |= {"none", "hand:w4", "hand:x4"}
    return plan, sorted(specs)


def main(stage="all", bands=("C-trace", "C-minall"), seeds=24, steps=400):
    OUT.mkdir(exist_ok=True)
    workers = int(os.environ.get("TCN_WORKERS", "16"))
    allplan, jobs = [], set()
    for band in bands:
        plan, specs = plan_for(band)
        allplan += plan
        jobs |= {(band, t, s) for t in tuple(evaltasks.LOO_TASKS) + CONTROLS
                 for s in specs}
    jobs = sorted(jobs)
    print(f"{len(jobs)} enumerations", flush=True)
    if stage in ("all", "enum"):
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_enum, jobs))
        for r in sorted(rows, key=lambda x: (x["band"], x["task"], x["spec"])):
            print(f"{r['band']:9s} {r['task']:18s} {r['spec']:22s} space "
                  f"{r['space_size']:8d} evaluated {r['evaluated']:8d} conforming "
                  f"{r['conforming']:5d} exhausted {r['exhausted']} "
                  f"{r['certificate']}", flush=True)
        (OUT / "heldout.json").write_text(json.dumps(
            {"plan": allplan, "rows": rows,
             "wall_seconds": time.perf_counter() - t0}, indent=1, sort_keys=True))
    if stage in ("all", "grad"):
        gjobs = [(b, t, s, seed, steps) for (b, t, s) in jobs
                 for seed in range(seeds)]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            recs = list(ex.map(_grad, gjobs))
        summary = {}
        for rec in recs:
            summary.setdefault((rec["band"], rec["task"], rec["spec"]), []).append(rec)
        out = []
        for (b, t, s), rows in sorted(summary.items()):
            accs = sorted(x["accuracy"] for x in rows)
            out.append({"band": b, "task": t, "spec": s, "seeds": len(rows),
                        "solved": sum(1 for x in rows if x["conformant"]),
                        "median_accuracy": accs[len(accs) // 2],
                        "mean_accuracy": sum(accs) / len(accs),
                        "best_accuracy": accs[-1],
                        "constant_baseline": evaltasks.constant_baseline(
                            evaltasks.COMPACT_TASKS[t]),
                        "random_baseline": 0.5})
        (OUT / "heldout_gradient.json").write_text(json.dumps(
            {"seeds": seeds, "steps": steps, "summary": out}, indent=1,
            sort_keys=True))
        for r in out:
            print(json.dumps(r, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
