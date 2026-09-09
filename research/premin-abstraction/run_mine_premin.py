"""Run the rule over every declared corpus variant and publish what it proposes.

`python run_mine_premin.py [variant ...]` -- one process per variant by
default.  Writes `out/proposal_<variant>.json` (the full ranked table, not just
the winner) and, for the corpora that feed an arm, publishes the rank-1
abstraction to a real `tcn.library.Library` under `library/`.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tcn.graph import Program
from tcn.library import Library
from tcn.operators import Registry
from tcn.types import BOOL, Value

import corpora
import mine
import mine_multi
import retention
from corpus import EARLIER_TASKS, examples_for

HERE = Path(__file__).parent
OUT = HERE / "out"
LIB = HERE / "library"


def load_bands():
    return json.loads((OUT / "corpora.json").read_text())


def build_variant(variant, data):
    """(corpus, task_of) for one declared variant. Values are frozen `Program`s."""
    corpus, task_of = {}, {}
    bands = {t["task"]: t["bands"] for t in data["tasks"]}
    if variant == "C-min":
        for task, gates in data["c_min"].items():
            corpus[task] = corpora.to_tcn(gates)
            task_of[task] = task
        return corpus, task_of
    if variant == "C-plus1-one":
        for task, b in bands.items():
            gates = b["plus1"]["first_in_order"]
            if gates is None:
                continue
            corpus[task] = corpora.to_tcn(gates)
            task_of[task] = task
        return corpus, task_of
    want = {"C-minall": ("min",), "C-plus1": ("plus1",), "C-trace": ("min", "plus1")}[variant]
    for task, b in bands.items():
        for label in want:
            for i, gates in enumerate(b[label]["programs"]):
                key = f"{task}#{label}{i:03d}"
                corpus[key] = corpora.to_tcn(gates)
                task_of[key] = task
    return corpus, task_of


def table_of_proposal(p):
    if p.holes > 4:
        return None
    return [[list(b), v] for b, v in mine.truth_table(p.canonical)]


def body_of(p):
    return [[n.candidates[0].operator.name, n.name, list(n.candidates[0].sources)]
            for n in p.canonical.nodes]


def is_maj(p):
    if p.holes != 3:
        return False
    tt = tuple(v for _, v in mine.truth_table(p.canonical))
    return tt == retention.MAJ3_TABLE


def one(variant):
    data = load_bands()
    corpus, task_of = build_variant(variant, data)
    fns = dict(EARLIER_TASKS)
    ex = {e: examples_for(fns[task_of[e]]) for e in corpus}
    t0 = time.perf_counter()
    props = mine_multi.propose(corpus, ex, task_of)
    wall = time.perf_counter() - t0

    table = []
    for p in props:
        row = p.row()
        row["truth_table"] = table_of_proposal(p)
        row["body"] = body_of(p)
        row["computes_maj3"] = is_maj(p)
        table.append(row)

    maj_rows = [r for r in table if r["computes_maj3"]]
    payload = {
        "variant": variant,
        "entries": len(corpus),
        "entries_per_task": {t: sum(1 for e in task_of.values() if e == t)
                             for t in sorted(set(task_of.values()))},
        "max_nodes": mine_multi.MAX_NODES, "max_holes": mine_multi.MAX_HOLES,
        "min_tasks": mine_multi.MIN_TASKS,
        "eligible": len(props), "mine_wall_seconds": wall,
        "rank1_computes_maj3": bool(table and table[0]["computes_maj3"]),
        "maj3_proposals": maj_rows,
        "best_maj3_rank": (min(r["rank"] for r in maj_rows) if maj_rows else None),
        "ranked": table,
    }
    (OUT / f"proposal_{variant}.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    head = table[0] if table else None
    print(variant, "entries", len(corpus), "eligible", len(props),
          "rank1", (head["body"] if head else None),
          "saving", (head["saving_bits"] if head else None),
          "maj3 rank", payload["best_maj3_rank"], "wall %.0fs" % wall, flush=True)
    return variant


def publish(variant, label, pick="rank1"):
    """Publish one abstraction from a mined table.

    `pick` is "rank1" (what the rule proposes) or "maj3" (the highest-ranked
    proposal that computes majority -- an *exploratory* choice made after
    seeing the ranking, used only to separate "the corpus does not contain it"
    from "the ranking does not pick it", and labelled as such everywhere).
    """
    data = json.loads((OUT / f"proposal_{variant}.json").read_text())
    if not data["ranked"]:
        print("nothing to publish for", variant)
        return
    if pick == "maj3":
        rows = [r for r in data["ranked"] if r["computes_maj3"]]
        if not rows:
            print("no maj3 proposal in", variant)
            return
        top = min(rows, key=lambda r: r["rank"])
    else:
        top = data["ranked"][0]
    corpus, task_of = build_variant(variant, load_bands())
    # rebuild the canonical program for the rank-1 digest
    canon = None
    for _e, program in corpus.items():
        for _root, _S, c, _h in mine.fragments(program, mine_multi.MAX_NODES,
                                               mine_multi.MAX_HOLES):
            if c.digest == top["digest"]:
                canon = c
                break
        if canon is not None:
            break
    assert canon is not None, top["digest"]
    fixture = [{k: Value.of(BOOL, v) for (k, _), v in zip(canon.inputs, bits)}
               for bits in itertools.product((False, True), repeat=len(canon.inputs))]
    entry = Library(LIB).publish(label, canon, Registry(), fixture=fixture,
                                 provenance={"rule": "mine_multi.propose", "corpus": variant,
                                             "pick": pick,
                                             "rank": top["rank"], "tasks": top["tasks"],
                                             "entries": top["entries"],
                                             "occurrences": top["occurrences"],
                                             "saving_bits": top["saving_bits"]})
    print("published", label, entry.reference, entry.operator, entry.nodes, "nodes",
          "rank", top["rank"], "maj3", top["computes_maj3"])
    return {"label": label, "variant": variant, "pick": pick, "rank": top["rank"],
            "tasks": top["tasks"], "entries": top["entries"],
            "occurrences": top["occurrences"], "saving_bits": top["saving_bits"],
            "reference": entry.reference,
            "digest": entry.digest, "operator": entry.operator, "nodes": entry.nodes,
            "description_bits": entry.description_bits,
            "execution_cost": entry.execution_cost,
            "fixture_cases": entry.fixture_cases, "truth_table": top["truth_table"],
            "body": top["body"], "computes_maj3": top["computes_maj3"]}


def main():
    OUT.mkdir(exist_ok=True)
    variants = sys.argv[1:] or list(corpora.VARIANTS)
    if variants == ["publish"]:
        published = {}
        for variant, label, pick in (("C-trace", "trace", "rank1"),
                                     ("C-minall", "minall", "rank1"),
                                     ("C-plus1", "plus1", "rank1"),
                                     ("C-plus1-one", "plus1one", "rank1"),
                                     ("C-trace", "trace_maj3", "maj3")):
            if (OUT / f"proposal_{variant}.json").exists():
                rec = publish(variant, label, pick)
                if rec:
                    published[label] = rec
        (OUT / "published.json").write_text(json.dumps(published, indent=2, sort_keys=True))
        return
    with ProcessPoolExecutor(max_workers=min(len(variants), 6)) as pool:
        list(pool.map(one, variants))


if __name__ == "__main__":
    main()
