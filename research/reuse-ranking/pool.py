"""One instrumented pooling pass; every objective is an aggregation of its output.

`mine_semantic.propose` (§46, unchanged, imported by §52) pools mined fragments
by `(arity, truth table)`, elects a representative per pool, rewrites every
occurrence to a call to it, verifies each rewrite against the task's rows, and
scores the pool by `total_baseline - rewritten_total - definition`.  That last
step throws away the per-entry decomposition, which is exactly what a different
aggregation needs.

`build()` below re-derives the identical pass and *keeps* the decomposition:
for every surviving class it records `s_e = b_e - r_e` for each corpus entry.
It changes nothing about the rule -- same fragments, same pooling key, same
representative election, same rewriting, same verification, same **default**
parameters `MAX_NODES=5, MAX_HOLES=4, MIN_TASKS=2` -- imported from
`mine_multi`, not restated.  They are exposed as arguments only so that
`run_sensitivity.py` can report the rule's sensitivity to them, as §52 §8 did.
**Every declared arm of this track uses the defaults**; nothing is tuned.

`check()` is integrity check IC1 of PREREGISTRATION.md: it runs
`mine_semantic.propose` on the same corpus and asserts, per class, exact integer
equality of `sum(s_e) - D` with the rule's own `saving_bits`, plus equality of
digest, tasks, entries and occurrences.  If IC1 fails the track stops.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import _paths  # noqa: F401

from tcn.operators import Registry
from tcn.types import BOOL, Value

import mine
import mine_semantic
from mine_multi import MAX_HOLES, MAX_NODES, MIN_TASKS, _conforms

import family


@dataclass
class Klass:
    """One pooled `(arity, truth table)` class, with its per-entry saving kept."""
    key: tuple
    digest: str
    canonical: object
    circuits: int
    nodes: int
    holes: int
    tasks: tuple
    entries: int
    occurrences: int
    definition_bits: int
    baseline_bits: int
    saving_by_entry: dict = field(default_factory=dict)   # entry -> s_e (hit entries only)
    occ_by_entry: dict = field(default_factory=dict)      # entry -> call sites
    rewritten: dict = field(default_factory=dict)         # entry -> rewritten Program

    @property
    def saving_bits(self):
        return sum(self.saving_by_entry.values()) - self.definition_bits

    def row(self):
        return {"digest": self.digest, "truth_table": [int(v) for v in self.key[1]],
                "arity": self.holes, "nodes": self.nodes,
                "circuits_pooled": self.circuits, "tasks": list(self.tasks),
                "entries": self.entries, "occurrences": self.occurrences,
                "definition_bits": self.definition_bits,
                "baseline_bits": self.baseline_bits,
                "saving_bits": self.saving_bits}


def build(family_name="maj", variant="C-minall", exclude=(), keep_rewritten=False,
          max_nodes=MAX_NODES, max_holes=MAX_HOLES, min_tasks=MIN_TASKS):
    """The instrumented pass.  Returns (classes, corpus, task_of, baseline)."""
    corpus, task_of = family.corpus_for(family_name, variant, exclude)
    fns = dict(family.FAMILIES[family_name])
    import evaltasks
    examples_by_entry = {e: evaltasks.examples_for(fns[task_of[e]]) for e in corpus}

    cache, pools = {}, {}
    for entry, program in corpus.items():
        for root, S, canon, holes in mine.fragments(program, max_nodes, max_holes):
            key = (len(canon.inputs), mine_semantic.truth_table(canon, cache))
            slot = pools.setdefault(key, {"reps": {}, "per_entry": {}})
            slot["reps"][canon.digest] = canon
            slot["per_entry"].setdefault(entry, []).append(
                {"root": root, "nodes": set(S), "holes": holes})

    baseline = {e: p.description_bits() for e, p in corpus.items()}
    total_baseline = sum(baseline.values())

    out = []
    for key, slot in pools.items():
        tasks = tuple(sorted({task_of[e] for e in slot["per_entry"]}))
        if len(tasks) < min_tasks:
            continue
        canon = min(slot["reps"].values(), key=lambda c: (len(c.nodes), c.digest))
        registry = Registry()
        name = registry.register_module(canon)
        if name != "module:" + canon.digest:
            continue
        sav, occ_by, rws = {}, {}, {}
        occurrences, hit_entries, ok = 0, 0, True
        for entry, program in corpus.items():
            occs = slot["per_entry"].get(entry)
            if not occs:
                continue
            chosen = mine.disjoint_occurrences(sorted(occs, key=lambda o: o["root"]))
            try:
                rw = mine.rewrite(program, chosen, name, registry)
            except (TypeError, ValueError, KeyError):
                ok = False
                break
            if not _conforms(rw, examples_by_entry[entry], registry):
                ok = False
                break
            sav[entry] = baseline[entry] - rw.description_bits()
            occ_by[entry] = len(chosen)
            if keep_rewritten:
                rws[entry] = rw
            occurrences += len(chosen)
            hit_entries += 1
        if not ok or occurrences < min_tasks:
            continue
        out.append(Klass(key=key, digest=canon.digest, canonical=canon,
                         circuits=len(slot["reps"]), nodes=len(canon.nodes),
                         holes=len(canon.inputs), tasks=tasks, entries=hit_entries,
                         occurrences=occurrences,
                         definition_bits=canon.description_bits(),
                         baseline_bits=total_baseline,
                         saving_by_entry=sav, occ_by_entry=occ_by, rewritten=rws))
    return out, corpus, task_of, baseline


def check(classes, family_name="maj", variant="C-minall", exclude=()):
    """IC1: the re-derivation must agree with the rule exactly."""
    corpus, task_of = family.corpus_for(family_name, variant, exclude)
    fns = dict(family.FAMILIES[family_name])
    import evaltasks
    ex = {e: evaltasks.examples_for(fns[task_of[e]]) for e in corpus}
    props = mine_semantic.propose(corpus, ex, task_of)
    mine_by = {p.key: p for p in props}
    ours_by = {k.key: k for k in classes}
    report = {"n_rule": len(props), "n_ours": len(classes),
              "keys_match": sorted(mine_by) == sorted(ours_by), "mismatches": []}
    for key, p in mine_by.items():
        k = ours_by.get(key)
        if k is None:
            report["mismatches"].append({"key_arity": key[0], "why": "missing"})
            continue
        for fld, a, b in (("saving_bits", p.saving_bits, k.saving_bits),
                          ("digest", p.digest, k.digest),
                          ("tasks", tuple(p.tasks), tuple(k.tasks)),
                          ("entries", p.entries, k.entries),
                          ("occurrences", p.occurrences, k.occurrences),
                          ("definition_bits", p.definition_bits, k.definition_bits),
                          ("nodes", p.nodes, k.nodes)):
            if a != b:
                report["mismatches"].append(
                    {"digest": p.digest, "field": fld, "rule": a, "ours": b})
    report["ic1_pass"] = bool(report["keys_match"] and not report["mismatches"])
    report["rule_rank1_digest"] = props[0].digest if props else None
    return report


def window_table(family_name="maj"):
    import retention
    if family_name == "maj":
        return retention.MAJ3_TABLE
    return tuple(family.d134(*b) for b in itertools.product((False, True), repeat=3))


def is_window(k, family_name="maj"):
    return bool(k.holes == 3 and tuple(bool(v) for v in k.key[1]) == window_table(family_name))


def fixture_for(canon):
    return [{key: Value.of(BOOL, v) for (key, _), v in zip(canon.inputs, bits)}
            for bits in itertools.product((False, True), repeat=len(canon.inputs))]
