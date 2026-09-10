"""Drive the two identity relations over this family's corpora and resolve the
pre-registered selection rules against the ranked tables.

Nothing about either rule is modified.  `mine_multi.propose` (identity =
`Program.digest`) and `mine_semantic.propose` (identity = `(arity, truth
table)`) are imported from `research/premin-abstraction` exactly as §46 left
them and exactly as §52 drove them; this module only supplies the corpus,
records the full ranked table, and resolves the *selection rules* fixed in
`PREREGISTRATION.md`:

    rank1           the rule's own proposal
    runnerup        the highest-ranked arity-4 class that does **not** compute
                    the family window -- the wrong-module control, defined by
                    rule before any table was read
    node_matched    the highest-ranked arity-4 non-window class whose
                    representative has the same node count as the window's
    best_window     the highest-ranked class that *does* compute the window,
                    whatever its rank; separates "the corpus does not contain
                    it" from "the ranking does not pick it"

The arity-4 restriction on the wrong-module rules is deliberate and is the
counterpart of §52's arity-3 restriction: the control must cost the search the
same as the module it controls for, or the comparison is confounded by space
size.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import _paths  # noqa: F401

from tcn.library import Library
from tcn.operators import Registry
from tcn.types import BOOL, Value

import mine
import mine_multi
import mine_semantic

import evaltasks
import family

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
LIB = HERE / "library"

WINDOW_ARITY = 4


def _body(canon):
    return [[n.candidates[0].operator.name, n.name, list(n.candidates[0].sources)]
            for n in canon.nodes]


def _table(canon):
    return [int(v) for _, v in mine.truth_table(canon)] if len(canon.inputs) <= 4 else None


def run(family_name, variant="C-trace", exclude=(), identity="semantic", label=None):
    """Mine one corpus under one identity relation; write and return the table."""
    corpus, task_of = family.corpus_for(family_name, variant, exclude)
    fns = dict(family.FAMILIES[family_name])
    ex = {e: evaltasks.examples_for(fns[task_of[e]]) for e in corpus}
    t0 = time.perf_counter()
    if identity == "semantic":
        props = mine_semantic.propose(corpus, ex, task_of)
    else:
        props = mine_multi.propose(corpus, ex, task_of)
    wall = time.perf_counter() - t0

    window = family.window_table(family_name)
    table = []
    for p in props:
        row = p.row()
        row["body"] = _body(p.canonical)
        row["truth_table"] = _table(p.canonical)
        row["arity"] = p.holes
        row["is_window"] = bool(p.holes == WINDOW_ARITY
                                and row["truth_table"] is not None
                                and tuple(bool(v) for v in row["truth_table"]) == window)
        table.append(row)

    payload = {
        "family": family_name, "variant": variant, "identity": identity,
        "excluded": sorted(exclude), "label": label,
        "entries": len(corpus), "tasks_in_corpus": sorted(set(task_of.values())),
        "eligible": len(props), "mine_wall_seconds": wall,
        "max_nodes": mine_semantic.MAX_NODES, "max_holes": mine_semantic.MAX_HOLES,
        "min_tasks": mine_semantic.MIN_TASKS,
        "rank1_is_window": bool(table and table[0]["is_window"]),
        "best_window_rank": min([r["rank"] for r in table if r["is_window"]], default=None),
        "window_classes": sum(1 for r in table if r["is_window"]),
        "ranked": table,
    }
    name = label or f"{family_name}_{variant}_{identity}" + (
        "_wo_" + "+".join(sorted(exclude)) if exclude else "")
    payload["name"] = name
    OUT.mkdir(exist_ok=True)
    (OUT / f"mined_{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def pick(payload, rule):
    rows = payload["ranked"]
    if not rows:
        return None
    if rule == "rank1":
        return rows[0]
    arn = [r for r in rows if r["arity"] == WINDOW_ARITY]
    if rule == "rank1_windowarity":
        return arn[0] if arn else None
    nonwin = [r for r in arn if not r["is_window"]]
    if rule == "runnerup":
        return nonwin[0] if nonwin else None
    if rule == "best_window":
        win = [r for r in rows if r["is_window"]]
        return win[0] if win else None
    if rule == "node_matched":
        win = [r for r in rows if r["is_window"]]
        if not win:
            return None
        same = [r for r in nonwin if r["nodes"] == win[0]["nodes"]]
        return same[0] if same else None
    raise ValueError(rule)


def rebuild(payload, digest):
    """The canonical `Program` behind one digest, rebuilt from the same corpus."""
    corpus, _ = family.corpus_for(payload["family"], payload["variant"],
                                  payload["excluded"])
    for _e, program in corpus.items():
        for _r, _S, canon, _h in mine.fragments(program, mine_semantic.MAX_NODES,
                                                mine_semantic.MAX_HOLES):
            if canon.digest == digest:
                return canon
    raise AssertionError(digest)


def fixture_for(canon):
    return [{k: Value.of(BOOL, v) for (k, _), v in zip(canon.inputs, bits)}
            for bits in itertools.product((False, True), repeat=len(canon.inputs))]


def publish(payload, row, label, lib=None):
    canon = rebuild(payload, row["digest"])
    entry = Library(lib or LIB).publish(label, canon, Registry(),
                                        fixture=fixture_for(canon), provenance={
        "rule": ("mine_semantic.propose" if payload["identity"] == "semantic"
                 else "mine_multi.propose"),
        "identity": payload["identity"], "family": payload["family"],
        "corpus": payload["variant"], "excluded": payload["excluded"],
        "rank": row["rank"], "tasks": row["tasks"], "entries": row["entries"],
        "saving_bits": row["saving_bits"], "arity": row["arity"],
        "is_window": row["is_window"], "preregistered": True})
    return {"label": label, "reference": entry.reference, "operator": entry.operator,
            "digest": entry.digest, "nodes": entry.nodes,
            "description_bits": entry.description_bits,
            "execution_cost": entry.execution_cost,
            "fixture_cases": entry.fixture_cases,
            "rank": row["rank"], "arity": row["arity"], "is_window": row["is_window"],
            "truth_table": row["truth_table"], "body": row["body"],
            "saving_bits": row["saving_bits"], "tasks": row["tasks"],
            "circuits_pooled": row.get("circuits_pooled"), "from": payload["name"]}
