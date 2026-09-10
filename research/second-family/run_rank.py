"""Score every objective on every corpus of the second family, and publish each
rank-1 class.

§54's `pool.build`, `context.build_context` and `objectives.rank` are imported
verbatim -- this file is §54's `run_rank.py` with the family swapped and O5
dropped for the reason declared in `PREREGISTRATION.md` §5.  The pool is built
once per corpus and every objective is an aggregation of that one pass, so no
arm differs from another in anything but the score.

Six corpora per band: the full six-task corpus and the five leave-one-out
corpora that remove one task entirely, at every length.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import _paths  # noqa: F401
import patches  # noqa: F401  -- re-points pool.is_window at this family

from tcn.library import Library
from tcn.operators import Registry

import context
import objectives
import pool

import armlib
import evaltasks
import family

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

OBJECTIVES = ("O1", "O2", "O3", "O4", "B1", "B2")
NEED = ("O3", "O4")


def publish(canon, row, corpus_name, lib):
    label = "cls_" + canon.digest[:12]
    if (lib / f"{label}.json").exists() or (lib / label).exists():
        try:
            Library(lib).load([label], registry=Registry(), policy="strict")
            return label
        except Exception:
            pass
    Library(lib).publish(label, canon, Registry(),
                         fixture=context.fixture_for(canon),
                         provenance={"rule": "mine_semantic pooling, unchanged",
                                     "identity": "(arity, truth table)",
                                     "track": "research/second-family",
                                     "selected_on": corpus_name,
                                     "arity": row["arity"], "nodes": row["nodes"],
                                     "truth_table": row["truth_table"],
                                     "preregistered": True})
    return label


def main(bands=("C-trace", "C-minall")):
    OUT.mkdir(exist_ok=True)
    for band in bands:
        lib = armlib.lib_for(band)
        lib.mkdir(exist_ok=True)
        corpora = [("full", ())] + [(f"wo_{t.split('_')[0]}", (t,))
                                    for t in evaltasks.LOO_TASKS]
        payload = {"band": band, "family": "w4",
                   "objectives": {k: objectives.NAME[k] for k in OBJECTIVES},
                   "corpora": {}, "integrity": {}}
        for name, exclude in corpora:
            t0 = time.perf_counter()
            classes, corpus, task_of, _base = pool.build("w4", band, exclude,
                                                         keep_rewritten=True)
            ic1 = pool.check(classes, "w4", band, exclude)
            ctx = context.build_context("w4", band, exclude, classes, corpus,
                                        task_of, need=NEED)
            rec = {"excluded": list(exclude), "entries": len(corpus),
                   "tasks_in_corpus": ctx["tasks_in_corpus"],
                   "eligible": len(classes),
                   "pool_wall_seconds": time.perf_counter() - t0, "tables": {}}
            for oid in OBJECTIVES:
                rows, ordered = objectives.rank(classes, oid, ctx)
                top = rows[0]
                lbl = publish(ordered[0].canonical, top, name, lib)
                rec["tables"][oid] = {
                    "rank1": dict(top, library_label=lbl),
                    "rank1_decided_by": objectives.rank1_decided_by(rows),
                    "best_window_rank": min([r["rank"] for r in rows
                                             if r["is_window"]], default=None),
                    "rank1_is_window": bool(top["is_window"]),
                    "ranked": rows}
                print(f"{band:9s} {name:8s} {oid}  rank1={top['digest'][:12]} "
                      f"arity={top['arity']} nodes={top['nodes']} "
                      f"tasks={len(top['tasks'])} window={top['is_window']} "
                      f"score={top['score']:.6g} "
                      f"window_rank={rec['tables'][oid]['best_window_rank']} "
                      f"decided_by={rec['tables'][oid]['rank1_decided_by']}",
                      flush=True)
            rec["ic1"] = ic1
            payload["corpora"][name] = rec
            payload["integrity"][name] = {
                "ic1_pass": ic1["ic1_pass"],
                "ic2_o1_matches_rule": (rec["tables"]["O1"]["rank1"]["digest"]
                                        == ic1["rule_rank1_digest"]),
                "rule_rank1_digest": ic1["rule_rank1_digest"]}
            print(f"  IC1 {ic1['ic1_pass']}  IC2 "
                  f"{payload['integrity'][name]['ic2_o1_matches_rule']}", flush=True)
        (OUT / f"ranked_{band}.json").write_text(json.dumps(payload, indent=1,
                                                            sort_keys=True))
        print("->", OUT / f"ranked_{band}.json", flush=True)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("C-trace", "C-minall"))
