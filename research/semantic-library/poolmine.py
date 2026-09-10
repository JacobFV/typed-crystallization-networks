"""Run both identity relations over a corpus, and pick modules by the rules
fixed in `PREREGISTRATION.md` rather than by inspection.

Nothing in the rule is modified.  `mine_multi.propose` (identity =
`Program.digest`) and `mine_semantic.propose` (identity = `(arity, truth table)`)
are imported from `research/premin-abstraction` exactly as §46 left them; this
module only drives them, records the full ranked table, and resolves the
pre-registered *selection rules* against it:

    rank-1                the rule's own proposal
    runner-up             the rank-1 arity-3 class that does **not** compute
                          the family's window function -- the wrong-module
                          control, defined by rule before the table was read
    node-matched          the rank-1 arity-3 non-window class whose
                          representative has the same node count as rank-1's

The arity-3 restriction is the budget constraint declared in the
pre-registration: a 4-ary module makes §44's tight space 64 161 792 programs,
which §46 measured at 8 184 wall-seconds for one arm.
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
import retention

import evaltasks
import family

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
LIB = HERE / "library"

WINDOW_TABLE = {
    "maj": retention.MAJ3_TABLE,
    "off": tuple(family.d134(*b) for b in itertools.product((False, True), repeat=3)),
}


def _body(canon):
    return [[n.candidates[0].operator.name, n.name, list(n.candidates[0].sources)]
            for n in canon.nodes]


def _table(canon):
    return [int(v) for _, v in mine.truth_table(canon)] if len(canon.inputs) <= 4 else None


def run(family_name, variant="C-minall", exclude=(), identity="semantic", label=None):
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

    window = WINDOW_TABLE[family_name]
    table = []
    for p in props:
        row = p.row()
        row["body"] = _body(p.canonical)
        row["truth_table"] = _table(p.canonical)
        row["arity"] = p.holes
        row["is_window"] = bool(p.holes == 3 and row["truth_table"] is not None
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
        "ranked": table,
    }
    OUT.mkdir(exist_ok=True)
    name = label or f"{family_name}_{variant}_{identity}" + (
        "_wo_" + "+".join(sorted(exclude)) if exclude else "")
    payload["name"] = name
    (OUT / f"mined_{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


# ------------------------------------------------------------------- selection

def pick(payload, rule):
    """Resolve a pre-registered selection rule against a ranked table."""
    rows = payload["ranked"]
    if not rows:
        return None
    if rule == "rank1":
        return rows[0]
    ar3 = [r for r in rows if r["arity"] == 3]
    if rule == "rank1_arity3":
        return ar3[0] if ar3 else None
    if rule == "rank2_arity3":
        # §44's arm 4b: the rule's *runner-up*, i.e. rank 2. Kept distinct from
        # `runnerup` below because the syntactic table's rank 1 is already a
        # non-majority fragment, so "top non-majority" collapses onto rank 1
        # there and would make arm 4b identical to arm 2.
        return ar3[1] if len(ar3) > 1 else None
    nonwin = [r for r in ar3 if not r["is_window"]]
    if rule == "runnerup":
        return nonwin[0] if nonwin else None
    if rule == "best_window":
        win = [r for r in rows if r["is_window"]]
        return win[0] if win else None
    if rule == "node_matched":
        top = rows[0]
        same = [r for r in nonwin if r["nodes"] == top["nodes"]]
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


def publish(payload, row, label):
    canon = rebuild(payload, row["digest"])
    fixture = [{k: Value.of(BOOL, v) for (k, _), v in zip(canon.inputs, bits)}
               for bits in itertools.product((False, True), repeat=len(canon.inputs))]
    entry = Library(LIB).publish(label, canon, Registry(), fixture=fixture, provenance={
        "rule": ("mine_semantic.propose" if payload["identity"] == "semantic"
                 else "mine_multi.propose"),
        "identity": payload["identity"], "family": payload["family"],
        "corpus": payload["variant"], "excluded": payload["excluded"],
        "rank": row["rank"], "tasks": row["tasks"], "entries": row["entries"],
        "saving_bits": row["saving_bits"], "arity": row["arity"],
        "is_window": row["is_window"],
        "preregistered": True})
    return {"label": label, "reference": entry.reference, "operator": entry.operator,
            "digest": entry.digest, "nodes": entry.nodes,
            "description_bits": entry.description_bits,
            "execution_cost": entry.execution_cost,
            "fixture_cases": entry.fixture_cases,
            "rank": row["rank"], "arity": row["arity"], "is_window": row["is_window"],
            "truth_table": row["truth_table"], "body": row["body"],
            "saving_bits": row["saving_bits"], "tasks": row["tasks"],
            "circuits_pooled": row.get("circuits_pooled"),
            "from": payload["name"]}
