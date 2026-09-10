"""The corpus variants this track compares.

Every variant is built over the *same* six earlier tasks as
`research/earned-abstraction/corpus.py` (imported, not restated) and in the
same basis.  They differ only in which conforming programs enter the corpus.

    C-min      one program per task: the certified minimum-length program the
               earned-abstraction solver returns first in its enumeration
               order.  This is exactly §44's corpus and is loaded from that
               track's `out/corpus_exact.json`, not re-derived.

    C-minall   *every* pruned program at the certified minimum length k_task.
               Still minimised, but no longer only one minimum: the tie-break
               is removed.

    C-plus1    every pruned program at length k_task + 1 -- genuinely
               non-minimal conforming programs.  Exhaustive where the budget
               allows, otherwise the declared sampler of
               `enumerate_programs.sample`; which one was used is recorded per
               task.

    C-trace    C-minall union C-plus1 -- "the programs enumerated on the way,
               not just the minimum".

    C-plus1-one  one program per task drawn from C-plus1, first in enumeration
               order.  A same-size control for C-min, so the multiset effect
               and the non-minimality effect can be told apart.

Per task and per length band the corpus is capped at `CAP` programs.  When the
enumeration exceeds the cap the retained subset is the deterministic
subsample "sort by sha256 of the canonical key, take the first CAP", which is
independent of enumeration order.  The uncapped counts are always reported
beside the capped ones.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from tcn.operators import Registry

import minimal
import enumerate_programs as E
from corpus import CORPUS_INPUTS, EARLIER_TASKS

HERE = Path(__file__).parent
OUT = HERE / "out"
EA_OUT = HERE.parent / "earned-abstraction" / "out"

CAP = 32
N = len(CORPUS_INPUTS)
VARIANTS = ("C-min", "C-minall", "C-plus1", "C-trace", "C-plus1-one")


def _hash(gates):
    return hashlib.sha256(repr(sorted(E.key_of(gates, [E.var_table(N, j) for j in range(N)], N))
                               ).encode()).hexdigest()


def subsample(gate_lists, cap=CAP):
    """Deterministic, enumeration-order-independent subsample of `cap` programs."""
    if len(gate_lists) <= cap:
        return list(gate_lists)
    ranked = sorted(gate_lists, key=_hash)
    return ranked[:cap]


def to_tcn(gates, registry=None):
    return minimal.to_program(N, [tuple(g) for g in gates], CORPUS_INPUTS, registry or Registry())


def min_lengths():
    """The certified minimum length per task, read from §44's artifact."""
    d = json.loads((EA_OUT / "corpus_exact.json").read_text())
    return {r["task"]: r["min_gates"] for r in d["report"] if r.get("solved")}


def c_min():
    """§44's corpus, loaded from that track's artifact rather than re-derived."""
    d = json.loads((EA_OUT / "corpus_exact.json").read_text())
    return {r["task"]: [tuple(g) for g in r["gates"]] for r in d["report"] if r.get("solved")}


def enumerate_band(fn, length, budget_seconds, trials, seed):
    """All pruned programs of `length` gates for `fn`, exhaustively or sampled."""
    tt = E.table_of(N, fn)
    t0 = time.perf_counter()
    sols, exhausted, expanded = E.exhaustive(N, tt, length)
    wall = time.perf_counter() - t0
    if wall <= budget_seconds:
        return {"length": length, "method": "exhaustive", "exhausted": bool(exhausted),
                "found": len(sols), "dfs_expanded": expanded, "wall_seconds": wall,
                "programs": sols}
    sols2, trials_run, hits = E.sample(N, tt, length, trials, seed)
    return {"length": length, "method": "sampled", "exhausted": False,
            "found": len(sols2), "trials": trials_run, "prefixes_hit": hits,
            "wall_seconds": time.perf_counter() - t0 + wall, "programs": sols2}


def verify_all(gate_lists, fn):
    r = Registry()
    for gates in gate_lists:
        p = to_tcn(gates, r)
        if not minimal.verify(p, fn, CORPUS_INPUTS, r):
            return False
    return True
