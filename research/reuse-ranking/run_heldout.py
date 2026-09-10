"""The decisive measurement: does each objective's rank-1 class help the task
that was removed from the corpus it was mined from?

The protocol is §52's, reused rather than rebuilt: the compact scaffold of
`research/semantic-library/evaltasks.py`, `tcn.search.enumerate_fit` with
`tolerance=1e-3`, `rank="order"` and `max_programs` above the space size so
every run is a **full sweep** and the certificate survives.  A module selected
on the corpus `C-minall ∖ t` is offered to `t`, which contributed no program at
any length to that corpus.

Beside every objective: the floor (`arm1_none`, the flat compact scaffold), the
ceiling (`arm3_authored`, the hand-authored `MAJ3`), and the two off-family
negative controls `H_par` and `H_d134`, which every majority-family arm must
leave at 0.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _paths  # noqa: F401

from tcn.library import Library
from tcn.operators import Registry
from tcn.search import candidate_counts, enumerate_fit, space_size

import evaltasks
import later
import objectives

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
LIB = HERE / "library"

LOO_TASKS = ("t1_maj_abc_xor_d", "t2_maj_abc_and_d", "t3_maj_bcd_or_a",
             "t4_maj_acd_xor_b", "t5_maj_abd_or_c")
CONTROLS = ("H_par", "H_d134")


def registry_for(spec):
    """(registry, module operator name) for `none`, `hand:<kind>` or `lib:<label>`."""
    r = Registry()
    if spec == "none":
        return r, None
    kind, name = spec.split(":", 1)
    if kind == "hand":
        return r, r.register_module(later.hand_authored(r, name))
    _, aliases = Library(LIB).load([name], registry=r, policy="strict")
    return r, aliases[name]


def _enum(job):
    task, spec = job
    fn = evaltasks.COMPACT_TASKS[task]
    r, module_name = registry_for(spec)
    program = evaltasks.compact_scaffold(r, module_name)
    examples = evaltasks.examples_for(fn)
    counts = candidate_counts(program)
    total = space_size(program)
    t0 = time.perf_counter()
    res = enumerate_fit(program, examples, evaltasks.SIGNALS, registry=r,
                        tolerance=.001, max_programs=1 << 24, rank="order")
    row = res.to_dict()
    row.update(task=task, spec=spec, module=module_name,
               candidates_per_node=list(counts), declared_space_size=total,
               constant_baseline=evaltasks.constant_baseline(fn),
               random_baseline=0.5,
               wall_seconds=time.perf_counter() - t0)
    if res.selections:
        row["module_on_output_path"] = later.module_on_output_path(
            program.harden(res.selections))
    return row


def main():
    OUT.mkdir(exist_ok=True)
    ranked = json.loads((OUT / "ranked.json").read_text())

    # which module each objective selects on each leave-one-out corpus
    plan, specs = [], set()
    for t in LOO_TASKS:
        cname = "wo_" + t.split("_")[0]
        for oid in objectives.ALL_IDS:
            top = ranked["corpora"][cname]["tables"][oid]["rank1"]
            spec = "lib:" + top["library_label"]
            plan.append({"objective": oid, "corpus": cname, "held_out": t,
                         "spec": spec, "digest": top["digest"],
                         "arity": top["arity"], "nodes": top["nodes"],
                         "tasks_in_class": len(top["tasks"]),
                         "is_window": top["is_window"],
                         "rank1_decided_by":
                             ranked["corpora"][cname]["tables"][oid]["rank1_decided_by"]})
            specs.add(spec)
    # the full-corpus selections too, for the in-sample column and the controls
    for oid in objectives.ALL_IDS:
        specs.add("lib:" + ranked["corpora"]["full"]["tables"][oid]["rank1"]["library_label"])
    specs |= {"none", "hand:maj", "hand:distractor"}

    jobs = sorted({(t, s) for t in LOO_TASKS + CONTROLS for s in sorted(specs)})
    workers = int(os.environ.get("TCN_WORKERS", "16"))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_enum, jobs))
    for r in sorted(rows, key=lambda x: (x["task"], x["spec"])):
        print(f"{r['task']:22s} {r['spec']:22s} space {r['space_size']:8d} "
              f"evaluated {r['evaluated']:8d} conforming {r['conforming']:5d} "
              f"exhausted {r['exhausted']} {r['certificate']}", flush=True)
    (OUT / "heldout.json").write_text(json.dumps(
        {"plan": plan, "rows": rows, "wall_seconds": time.perf_counter() - t0},
        indent=1, sort_keys=True))
    print("->", OUT / "heldout.json")


if __name__ == "__main__":
    main()
