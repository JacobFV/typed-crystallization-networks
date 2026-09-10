"""Amendment 1: run the class B1's tie-break did **not** pick.

On the `C-trace` band the frequency baseline's score ties at the top on six
classes; the tie-break `(-nodes, digest)` narrows that to two three-node
classes and then picks the window on hex-digest order.  This offers the other
one to the same five held-out tasks under the identical protocol, so the
write-up can say whether B1's `5 of 5` is frequency selecting or a two-way
lexicographic coin flip.

Exploratory.  It changes no primary verdict.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import _paths  # noqa: F401
import patches  # noqa: F401

from tcn.library import Library
from tcn.operators import Registry
from tcn.search import candidate_counts, enumerate_fit, space_size

import context
import pool

import armlib
import evaltasks

OUT = Path(__file__).resolve().parent / "out"
BAND = "C-trace"


def tie_alternative(exclude):
    """The runner-up under B1's own tie-break on one leave-one-out corpus."""
    classes, _c, _t, _b = pool.build("w4", BAND, exclude)
    scored = sorted(classes, key=lambda k: (-len(k.tasks), -k.nodes, k.digest))
    top = scored[0]
    tied = [k for k in scored if len(k.tasks) == len(top.tasks)]
    return top, (tied[1] if len(tied) > 1 else None)


def _enum(job):
    task, label = job
    r = Registry()
    _, aliases = Library(armlib.lib_for(BAND)).load([label], registry=r,
                                                    policy="strict")
    program = evaltasks.compact_scaffold(r, aliases[label])
    fn = evaltasks.COMPACT_TASKS[task]
    t0 = time.perf_counter()
    res = enumerate_fit(program, evaltasks.examples_for(fn), evaltasks.SIGNALS,
                        registry=r, tolerance=.001, max_programs=1 << 24,
                        rank="order")
    row = res.to_dict()
    row.update(task=task, label=label, band=BAND,
               candidates_per_node=list(candidate_counts(program)),
               declared_space_size=space_size(program),
               constant_baseline=evaltasks.constant_baseline(fn),
               random_baseline=0.5, wall_seconds=time.perf_counter() - t0)
    return row


def main():
    OUT.mkdir(exist_ok=True)
    lib = armlib.lib_for(BAND)
    plan, jobs = [], set()
    for t in evaltasks.LOO_TASKS:
        top, alt = tie_alternative((t,))
        if alt is None:
            plan.append({"held_out": t, "no_alternative": True})
            continue
        label = "alt_" + alt.digest[:12]
        if not (lib / f"{label}.json").exists():
            Library(lib).publish(label, alt.canonical, Registry(),
                                 fixture=context.fixture_for(alt.canonical),
                                 provenance={"track": "research/second-family",
                                             "amendment": 1,
                                             "role": "B1 tie-break alternative",
                                             "exploratory": True})
        plan.append({"held_out": t, "b1_pick": top.digest,
                     "b1_pick_is_window": pool.is_window(top, "w4"),
                     "alt": alt.digest, "alt_is_window": pool.is_window(alt, "w4"),
                     "alt_nodes": alt.nodes, "alt_arity": alt.holes,
                     "tasks_tied": len(top.tasks), "label": label})
        jobs.add((t, label))
    with ProcessPoolExecutor(max_workers=int(os.environ.get("TCN_WORKERS", "8"))) as ex:
        rows = list(ex.map(_enum, sorted(jobs)))
    helps = 0
    for p in plan:
        if p.get("no_alternative"):
            print(f"{p['held_out']:20s} no alternative in the tie group", flush=True)
            continue
        r = next(x for x in rows if x["task"] == p["held_out"]
                 and x["label"] == p["label"])
        helps += int(r["conforming"] > 0)
        print(f"{p['held_out']:20s} B1 picked {p['b1_pick'][:12]} "
              f"(window={p['b1_pick_is_window']}), alternative {p['alt'][:12]} "
              f"(window={p['alt_is_window']}) -> conforming {r['conforming']} "
              f"exhausted {r['exhausted']} {r['certificate']}", flush=True)
    print(f"B1's tie-break alternative helps {helps} of 5", flush=True)
    (OUT / "b1_alt.json").write_text(json.dumps(
        {"exploratory": True, "amendment": 1, "band": BAND, "plan": plan,
         "rows": rows, "helps_of_5": helps}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
