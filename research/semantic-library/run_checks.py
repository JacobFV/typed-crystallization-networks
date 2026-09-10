"""The two operationalisations of falsification criterion F3, plus the cost
accounting, computed from the mined tables rather than asserted.

F3(a)  Does the identical pooled rule, run on a family whose shared structure is
       **not** majority, propose that family's *own* window -- or the *same*
       fragment as the majority family?  The same fragment from both families
       would mean the ranking is reading the fragment enumerator, not the corpus.

F3(b)  Does replacing R4's MDL saving by the raw pooled occurrence count leave
       the rank-1 class unchanged?  If it does, the "selection" is frequency.

Also reported: how large the rank-1 pool is, whether it is the *largest* pool,
and what pooling costs against digest identity on the same corpus.
"""
from __future__ import annotations

import json
from pathlib import Path

import _paths  # noqa: F401

import family

OUT = Path(__file__).resolve().parent / "out"


def load(name):
    return json.loads((OUT / f"mined_{name}.json").read_text())


def summarise(p):
    rows = p["ranked"]
    top = rows[0] if rows else None
    by_occ = sorted(rows, key=lambda r: (-r["occurrences"], r["rank"]))
    by_pool = sorted(rows, key=lambda r: (-(r.get("circuits_pooled") or 1), r["rank"]))
    return {
        "name": p["name"], "family": p["family"], "identity": p["identity"],
        "variant": p["variant"], "excluded": p["excluded"],
        "entries": p["entries"], "eligible": p["eligible"],
        "mine_wall_seconds": p["mine_wall_seconds"],
        "rank1_is_window": p["rank1_is_window"],
        "best_window_rank": p["best_window_rank"],
        "rank1": None if top is None else {
            "digest": top["digest"], "arity": top["arity"], "nodes": top["nodes"],
            "truth_table": top["truth_table"], "body": top["body"],
            "occurrences": top["occurrences"], "saving_bits": top["saving_bits"],
            "circuits_pooled": top.get("circuits_pooled"),
            "tasks": top["tasks"], "entries": top["entries"]},
        "f3b_rank1_by_occurrence": None if not rows else {
            "digest": by_occ[0]["digest"], "rank_under_mdl": by_occ[0]["rank"],
            "arity": by_occ[0]["arity"], "occurrences": by_occ[0]["occurrences"],
            "is_window": by_occ[0]["is_window"],
            "truth_table": by_occ[0]["truth_table"]},
        "f3b_unchanged": bool(rows and by_occ[0]["digest"] == top["digest"]),
        "largest_pool": None if not rows else {
            "digest": by_pool[0]["digest"], "rank_under_mdl": by_pool[0]["rank"],
            "circuits_pooled": by_pool[0].get("circuits_pooled"),
            "is_window": by_pool[0]["is_window"]},
        "rank1_is_largest_pool": bool(rows and by_pool[0]["digest"] == top["digest"]),
    }


def main():
    names = ["maj_minall_syn", "maj_minall_sem", "maj_trace_sem",
             "off_minall_syn", "off_minall_sem"]
    names += [f"maj_minall_sem_wo_{t.split('_')[0]}" for t in
              ("t1", "t2", "t3", "t4", "t5")]
    names += [f"maj_minall_syn_wo_{t.split('_')[0]}" for t in
              ("t1", "t2", "t3", "t4", "t5")]
    out = {}
    for n in names:
        path = OUT / f"mined_{n}.json"
        if not path.exists():
            continue
        out[n] = summarise(load(n))

    maj = out.get("maj_minall_sem", {}).get("rank1")
    off = out.get("off_minall_sem", {}).get("rank1")
    f3a = {
        "maj_rank1_digest": maj and maj["digest"],
        "off_rank1_digest": off and off["digest"],
        "same_fragment": bool(maj and off and maj["digest"] == off["digest"]),
        "same_truth_table": bool(maj and off and maj["arity"] == off["arity"]
                                 and maj["truth_table"] == off["truth_table"]),
        "maj_rank1_truth_table": maj and maj["truth_table"],
        "off_rank1_truth_table": off and off["truth_table"],
        "off_window_truth_table": [int(v) for v in family.WINDOW_TABLE_OFF]
        if hasattr(family, "WINDOW_TABLE_OFF") else None,
        "off_rank1_is_off_window": out.get("off_minall_sem", {}).get("rank1_is_window"),
    }

    # cost: pooling against digest identity on the same corpus
    cost = {}
    for a, b in (("maj_minall_syn", "maj_minall_sem"),
                 ("off_minall_syn", "off_minall_sem")):
        if a in out and b in out:
            cost[b] = {"syntactic_mine_seconds": out[a]["mine_wall_seconds"],
                       "semantic_mine_seconds": out[b]["mine_wall_seconds"],
                       "ratio": out[b]["mine_wall_seconds"] / out[a]["mine_wall_seconds"],
                       "syntactic_eligible": out[a]["eligible"],
                       "semantic_eligible": out[b]["eligible"]}

    payload = {"per_corpus": out, "F3a_cross_family": f3a, "mining_cost": cost}
    (OUT / "checks.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"F3a": f3a, "cost": cost}, indent=2, sort_keys=True))
    for n, v in out.items():
        print(n, "rank1_is_window", v["rank1_is_window"], "best_window_rank",
              v["best_window_rank"], "F3b unchanged", v["f3b_unchanged"],
              "rank1 is largest pool", v["rank1_is_largest_pool"])


if __name__ == "__main__":
    main()
