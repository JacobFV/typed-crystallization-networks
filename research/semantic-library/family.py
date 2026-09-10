"""The two task families and the corpora mined from them.

The **majority family** is `research/earned-abstraction/corpus.py`'s six tasks,
imported unchanged.  Its `C-minall` / `C-trace` bands are *loaded* from §46's
`research/premin-abstraction/out/corpora.json`, not re-derived, so the corpus is
bit-identical to the one §46's exploratory measurement used.

The **off-family** `F'` is the same six task *shapes* with `maj` replaced by
`D134`, the §44 distractor.  Its bands are built here with §46's own
`enumerate_programs` / `corpora` machinery: the same basis, the same pruning
rule, the same cap of 32, the same deterministic sha256 subsample, and the same
in-`tcn` verification of every retained program against its full truth table.

`corpus_for(family, exclude=...)` is the only thing this track adds: it drops a
task from a corpus *entirely*, which is what a held-out task means here.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import _paths  # noqa: F401

from tcn.operators import Registry

import corpora
import enumerate_programs as E
import minimal
import run_mine_premin as RMP
from corpus import CORPUS_INPUTS, EARLIER_TASKS

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
N = len(CORPUS_INPUTS)


# ------------------------------------------------------------------ the families

def maj(a, b, c):
    return (int(a) + int(b) + int(c)) >= 2


def d134(a, b, c):
    """Truth table 134 -- `later.MINIMAL_BODIES["distractor"]`, same size and arity."""
    return bool(((bool(a) != bool(b)) != (bool(c) and (bool(a) or bool(b)))))


def _shapes(w):
    """The six task shapes of `corpus.py`, over an arbitrary 3-ary window `w`."""
    return (
        ("s1_w_abc_xor_d", lambda a, b, c, d: w(a, b, c) != bool(d)),
        ("s2_w_abc_and_d", lambda a, b, c, d: w(a, b, c) and bool(d)),
        ("s3_w_bcd_or_a", lambda a, b, c, d: w(b, c, d) or bool(a)),
        ("s4_w_acd_xor_b", lambda a, b, c, d: w(a, c, d) != bool(b)),
        ("s5_w_abd_or_c", lambda a, b, c, d: w(a, b, d) or bool(c)),
        ("s6_w_abc_xor_w_bcd", lambda a, b, c, d: w(a, b, c) != w(b, c, d)),
    )


OFF_TASKS = _shapes(d134)
MAJ_TASKS = EARLIER_TASKS

FAMILIES = {"maj": MAJ_TASKS, "off": OFF_TASKS}


# ------------------------------------------------------------------ off-family bands

def build_off_bands(cap=corpora.CAP, budget_seconds=1800.0, want=("min",)):
    """Bands for F', in §46's format and by §46's code.

    Only the `min` band is built by default, because `C-minall` is this track's
    declared primary corpus and it is the only F' corpus any declared arm uses.
    The `C-plus1` band was started once and **abandoned**: exhausting length 6
    over four inputs costs ~166.5M DFS nodes per task (§46's measured figure),
    and no arm reads it.  That abandonment is recorded here rather than in a
    footnote.
    """
    tasks = []
    t_all = time.perf_counter()
    for name, fn in OFF_TASKS:
        tt = E.table_of(N, fn)
        k, _gates, mstats = minimal.scaffold_min(N, tt, max_len=6)
        assert k is not None, name
        bands = {}
        for label, length in [(l, k if l == "min" else k + 1) for l in want]:
            t0 = time.perf_counter()
            sols, exhausted, expanded = E.exhaustive(N, tt, length)
            wall = time.perf_counter() - t0
            capped = corpora.subsample(sols, cap)
            r = Registry()
            ok = all(minimal.verify(minimal.to_program(N, [tuple(g) for g in gs],
                                                       CORPUS_INPUTS, r), fn,
                                    CORPUS_INPUTS, r)
                     for gs in capped)
            bands[label] = {"length": length, "method": "exhaustive",
                            "exhausted": bool(exhausted), "found": len(sols),
                            "uncapped_found": len(sols), "capped": len(capped),
                            "dfs_expanded": expanded, "wall_seconds": wall,
                            "verified_in_tcn": bool(ok),
                            "first_in_order": (sols[0] if sols else None),
                            "programs": capped}
        tasks.append({"task": name, "min_gates": k, "min_certificate": mstats, "bands": bands})
    payload = {"cap": cap, "inputs": list(CORPUS_INPUTS), "budget_seconds": budget_seconds,
               "family": "off", "window": "D134", "tasks": tasks,
               "wall_seconds": time.perf_counter() - t_all}
    OUT.mkdir(exist_ok=True)
    (OUT / "off_corpora.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def load_bands(family):
    if family == "maj":
        return RMP.load_bands()
    return json.loads((OUT / "off_corpora.json").read_text())


# ------------------------------------------------------------------ corpora

def corpus_for(family, variant="C-minall", exclude=()):
    """(corpus, task_of) with `exclude`d tasks removed **entirely**.

    A held-out task contributes no program, at any length, to the corpus the
    rule mines -- that is the whole point of the held-out protocol.
    """
    data = load_bands(family)
    if family == "maj":
        corpus, task_of = RMP.build_variant(variant, data)
    else:
        corpus, task_of = {}, {}
        want = {"C-minall": ("min",), "C-plus1": ("plus1",),
                "C-trace": ("min", "plus1")}[variant]
        for t in data["tasks"]:
            for label in want:
                for i, gates in enumerate(t["bands"][label]["programs"]):
                    key = f"{t['task']}#{label}{i:03d}"
                    corpus[key] = corpora.to_tcn(gates)
                    task_of[key] = t["task"]
    drop = set(exclude)
    corpus = {e: p for e, p in corpus.items() if task_of[e] not in drop}
    task_of = {e: t for e, t in task_of.items() if t not in drop}
    return corpus, task_of
