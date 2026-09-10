"""The seven ranking objectives of PREREGISTRATION.md §5.

Every one of them is an aggregation of the *same* instrumented pooling pass
(`pool.build`).  Nothing about the pool -- fragments, identity relation,
representative election, rewriting, verification, `MAX_NODES/MAX_HOLES/MIN_TASKS`
-- differs between arms.  Only the score and the ordering differ.

O1-O5 share `mine_semantic.propose`'s own tie-break with the score swapped, so
that any change in rank 1 is attributable to the score and not to the tie-break;
§52 observed that promoting `tasks` above `saving` in the tie-break alone would
fix the ordering, and that is deliberately not what any arm here does.
"""
from __future__ import annotations

import _paths  # noqa: F401

import pool

MDL_IDS = ("O1", "O2", "O3", "O4", "O5")
FREQ_IDS = ("B1", "B2")
ALL_IDS = MDL_IDS + FREQ_IDS

NAME = {
    "O1": "incumbent: description_bits summed over entries",
    "O2": "breadth-weighted: saving x distinct tasks",
    "O3": "per-task mean of per-entry saving, over all corpus tasks",
    "O4": "in-corpus leave-one-out cross-validated saving",
    "O5": "execution-cost-aware, measured primitive operations",
    "B1": "frequency baseline: number of distinct tasks",
    "B2": "frequency baseline: number of occurrences",
}


# --------------------------------------------------------------------- scores

def o1(k, ctx):
    return float(k.saving_bits)


def o2(k, ctx):
    return float(len(k.tasks) * sum(k.saving_by_entry.values()) - k.definition_bits)


def o3(k, ctx):
    """Mean over ALL corpus tasks of the mean per-entry saving inside that task.

    A task where the class does not occur contributes 0, so narrowness is
    penalised rather than rewarded; the inner mean removes the entry-count
    imbalance between tasks that O1 is sensitive to.
    """
    tasks = ctx["tasks_in_corpus"]
    per_task = []
    for t in tasks:
        ents = ctx["entries_of_task"][t]
        if not ents:
            continue
        per_task.append(sum(k.saving_by_entry.get(e, 0) for e in ents) / len(ents))
    return sum(per_task) / len(per_task) - k.definition_bits


def o4(k, ctx):
    """Mean over folds of the class's saving on the fold's *withheld* task.

    `ctx["cv"][u]` maps a pooling key to `(sum of s_e over u's entries, D)` in a
    pool mined from the corpus **minus u**; entries of `u` were withheld from
    that pool's derivation and from its scoring.  A key absent from the reduced
    pool scores 0 for that fold.
    """
    tasks = ctx["tasks_in_corpus"]
    tot = 0.0
    for u in tasks:
        hit = ctx["cv"].get(u, {}).get(k.key)
        tot += 0.0 if hit is None else float(hit[0] - hit[1])
    return tot / len(tasks)


def o5(k, ctx):
    """Measured executed primitive operations saved, definition charged once."""
    return float(ctx["o5_score"][k.key])


def b1(k, ctx):
    return float(len(k.tasks))


def b2(k, ctx):
    return float(k.occurrences)


SCORE = {"O1": o1, "O2": o2, "O3": o3, "O4": o4, "O5": o5, "B1": b1, "B2": b2}


# ---------------------------------------------------------------------- rank

def rank(classes, oid, ctx):
    """The full ranked table under one objective.  Returns a list of dicts."""
    fn = SCORE[oid]
    scored = [(fn(k, ctx), k) for k in classes]
    if oid in FREQ_IDS:
        key = lambda sk: (-sk[0], -sk[1].nodes, sk[1].digest)          # noqa: E731
    else:
        key = lambda sk: (-sk[0], -len(sk[1].tasks), -sk[1].occurrences,  # noqa: E731
                          -sk[1].nodes, sk[1].digest)
    scored.sort(key=key)
    rows = []
    for i, (s, k) in enumerate(scored):
        r = k.row()
        r.update(objective=oid, score=s, rank=i + 1,
                 is_window=pool.is_window(k, ctx.get("family", "maj")))
        rows.append(r)
    return rows, [k for _s, k in scored]


def rank1_decided_by(rows):
    """Was rank 1 settled by the score, or by the tie-break?"""
    if len(rows) < 2:
        return "single_class"
    return "score" if rows[0]["score"] != rows[1]["score"] else "tie_break"
