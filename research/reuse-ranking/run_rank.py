"""Score every objective on every corpus, and publish each rank-1 class.

Six corpora: the full six-task `C-minall` and the five leave-one-out corpora
that remove one task entirely, at every length.  Seven objectives.  The pool is
built once per corpus and every objective is an aggregation of that one pass, so
no arm differs from another in anything but the score.

Writes `out/ranked.json` (the full ranked table under every objective, on every
corpus, with the integrity checks) and publishes each distinct selected class
into `library/` as a real `tcn.library` entry with a full-truth-table fixture.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import _paths  # noqa: F401

from tcn.library import Library
from tcn.operators import Registry

import context
import objectives
import pool

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
LIB = HERE / "library"

LOO_TASKS = ("t1_maj_abc_xor_d", "t2_maj_abc_and_d", "t3_maj_bcd_or_a",
             "t4_maj_acd_xor_b", "t5_maj_abd_or_c")


def label_for(digest):
    return "cls_" + digest[:12]


def publish(canon, row, corpus_name):
    label = label_for(canon.digest)
    if (LIB / f"{label}.json").exists() or (LIB / label).exists():
        try:
            r = Registry()
            Library(LIB).load([label], registry=r, policy="strict")
            return label
        except Exception:
            pass
    Library(LIB).publish(label, canon, Registry(),
                         fixture=context.fixture_for(canon),
                         provenance={"rule": "mine_semantic pooling, unchanged",
                                     "identity": "(arity, truth table)",
                                     "track": "research/reuse-ranking",
                                     "selected_on": corpus_name,
                                     "arity": row["arity"], "nodes": row["nodes"],
                                     "truth_table": row["truth_table"],
                                     "preregistered": True})
    return label


def main():
    OUT.mkdir(exist_ok=True)
    LIB.mkdir(exist_ok=True)
    corpora = [("full", ())] + [(f"wo_{t.split('_')[0]}", (t,)) for t in LOO_TASKS]
    payload = {"objectives": {k: objectives.NAME[k] for k in objectives.ALL_IDS},
               "corpora": {}, "integrity": {}}
    for name, exclude in corpora:
        t0 = time.perf_counter()
        classes, corpus, task_of, _base = pool.build("maj", "C-minall", exclude,
                                                     keep_rewritten=True)
        ic1 = pool.check(classes, "maj", "C-minall", exclude)
        ctx = context.build_context("maj", "C-minall", exclude, classes, corpus, task_of)
        rec = {"excluded": list(exclude), "entries": len(corpus),
               "tasks_in_corpus": ctx["tasks_in_corpus"], "eligible": len(classes),
               "pool_wall_seconds": time.perf_counter() - t0,
               "o5_detail": ctx["o5_detail"], "tables": {}}
        for oid in objectives.ALL_IDS:
            rows, ordered = objectives.rank(classes, oid, ctx)
            top = rows[0]
            lbl = publish(ordered[0].canonical, top, name)
            rec["tables"][oid] = {
                "rank1": dict(top, library_label=lbl),
                "rank1_decided_by": objectives.rank1_decided_by(rows),
                "best_window_rank": min([r["rank"] for r in rows if r["is_window"]],
                                        default=None),
                "rank1_is_window": bool(top["is_window"]),
                "ranked": rows,
            }
            print(f"{name:8s} {oid}  rank1={top['digest'][:12]} "
                  f"arity={top['arity']} nodes={top['nodes']} "
                  f"tasks={len(top['tasks'])} window={top['is_window']} "
                  f"score={top['score']:.6g} "
                  f"window_rank={rec['tables'][oid]['best_window_rank']} "
                  f"decided_by={rec['tables'][oid]['rank1_decided_by']}", flush=True)
        rec["ic1"] = ic1
        payload["corpora"][name] = rec
        payload["integrity"][name] = {
            "ic1_pass": ic1["ic1_pass"],
            "ic2_o1_matches_rule": (rec["tables"]["O1"]["rank1"]["digest"]
                                    == ic1["rule_rank1_digest"]),
            "rule_rank1_digest": ic1["rule_rank1_digest"]}
        print(f"  IC1 {ic1['ic1_pass']}  IC2 {payload['integrity'][name]['ic2_o1_matches_rule']}",
              flush=True)
    (OUT / "ranked.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    print("->", OUT / "ranked.json")


if __name__ == "__main__":
    main()
