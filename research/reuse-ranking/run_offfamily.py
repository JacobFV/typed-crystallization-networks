"""EXPLORATORY replication on the second family, added after the primary arms.

This is **Amendment 1** to `PREREGISTRATION.md` (it adds to §6; it replaces
nothing and can change no primary verdict).  §52 disclosed as a limitation that
"the family count is one, plus one off-family corpus".  The off-family corpus
`F'` -- §52's six task shapes with the majority window replaced by the §44
distractor `D134`, 117 entries over six tasks, every entry verified in `tcn` --
is already on disk, so the identical leave-one-out protocol can be run on it for
the cost of five mines and a few enumerations.

It was written and run **after** the majority-family table was complete, so it
is exploratory in the strict sense and is labelled as such everywhere.  Its
outcome is reported whichever way it falls.

Negative controls here run the other way from §52's: an `F'` module is offered
to the majority tasks `t1`-`t5`, where it must score 0, and to `H_par`.
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

import context
import evaltasks
import family
import objectives
import pool

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
LIB = HERE / "library_off"

OFF = dict(family.OFF_TASKS)
OFF_LOO = ("s1_w_abc_xor_d", "s2_w_abc_and_d", "s3_w_bcd_or_a",
           "s4_w_acd_xor_b", "s5_w_abd_or_c")
# cross-family negative controls: majority tasks, plus parity
CONTROLS = {"t1_maj_abc_xor_d": evaltasks.COMPACT_TASKS["t1_maj_abc_xor_d"],
            "t3_maj_bcd_or_a": evaltasks.COMPACT_TASKS["t3_maj_bcd_or_a"],
            "H_par": evaltasks.COMPACT_TASKS["H_par"]}
TASK_FN = dict(OFF, **CONTROLS)


def d134_authored(r):
    """The hand-authored `D134` -- the ceiling for `F'`, as `MAJ3` is for `maj`."""
    import later
    return r.register_module(later.hand_authored(r, "distractor"))


def registry_for(spec):
    r = Registry()
    if spec == "none":
        return r, None
    kind, name = spec.split(":", 1)
    if kind == "hand":
        return r, d134_authored(r)
    _, aliases = Library(LIB).load([name], registry=r, policy="strict")
    return r, aliases[name]


def _enum(job):
    task, spec = job
    fn = TASK_FN[task]
    r, module_name = registry_for(spec)
    program = evaltasks.compact_scaffold(r, module_name)
    t0 = time.perf_counter()
    res = enumerate_fit(program, evaltasks.examples_for(fn), evaltasks.SIGNALS,
                        registry=r, tolerance=.001, max_programs=1 << 24, rank="order")
    row = res.to_dict()
    row.update(task=task, spec=spec, module=module_name,
               candidates_per_node=list(candidate_counts(program)),
               declared_space_size=space_size(program),
               constant_baseline=evaltasks.constant_baseline(fn),
               random_baseline=0.5, wall_seconds=time.perf_counter() - t0)
    return row


def main():
    OUT.mkdir(exist_ok=True)
    LIB.mkdir(exist_ok=True)
    tables, specs, plan = {}, {"none", "hand:d134"}, []
    for held in OFF_LOO:
        name = "wo_" + held.split("_")[0]
        classes, corpus, task_of, _ = pool.build("off", "C-minall", (held,),
                                                 keep_rewritten=True)
        ic1 = pool.check(classes, "off", "C-minall", (held,))
        ctx = context.build_context("off", "C-minall", (held,), classes, corpus, task_of)
        rec = {"entries": len(corpus), "eligible": len(classes), "ic1": ic1["ic1_pass"],
               "rule_rank1": ic1["rule_rank1_digest"], "tables": {}}
        for oid in objectives.ALL_IDS:
            rows, ordered = objectives.rank(classes, oid, ctx)
            top = rows[0]
            label = "off_" + top["digest"][:12]
            if not (LIB / f"{label}.json").exists():
                try:
                    Library(LIB).publish(label, ordered[0].canonical, Registry(),
                                         fixture=context.fixture_for(ordered[0].canonical),
                                         provenance={"track": "research/reuse-ranking",
                                                     "family": "off (D134)",
                                                     "exploratory": True,
                                                     "selected_on": name})
                except Exception:
                    pass
            spec = "lib:" + label
            specs.add(spec)
            rec["tables"][oid] = {"rank1": dict(top, library_label=label),
                                  "best_window_rank": min([r["rank"] for r in rows
                                                           if r["is_window"]], default=None)}
            plan.append({"objective": oid, "corpus": name, "held_out": held,
                         "spec": spec, "digest": top["digest"],
                         "is_window": top["is_window"], "arity": top["arity"],
                         "tasks_in_class": len(top["tasks"])})
            print(f"{name:8s} {oid} rank1={top['digest'][:12]} arity={top['arity']} "
                  f"tasks={len(top['tasks'])} window={top['is_window']} "
                  f"window_rank={rec['tables'][oid]['best_window_rank']}", flush=True)
        tables[name] = rec
        print(f"  IC1 {ic1['ic1_pass']}", flush=True)

    jobs = sorted({(t, s) for t in tuple(OFF_LOO) + tuple(CONTROLS) for s in sorted(specs)})
    with ProcessPoolExecutor(max_workers=int(os.environ.get("TCN_WORKERS", "12"))) as ex:
        rows = list(ex.map(_enum, jobs))
    for r in sorted(rows, key=lambda x: (x["task"], x["spec"])):
        print(f"{r['task']:20s} {r['spec']:22s} space {r['space_size']:8d} "
              f"conforming {r['conforming']:5d} exhausted {r['exhausted']} "
              f"{r['certificate']}", flush=True)
    (OUT / "offfamily.json").write_text(json.dumps(
        {"exploratory": True, "amendment": 1, "tables": tables, "plan": plan,
         "rows": rows}, indent=1, sort_keys=True))
    print("->", OUT / "offfamily.json")


if __name__ == "__main__":
    main()
