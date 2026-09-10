"""The same rule with one thing changed: identity is *semantic*, not structural.

§44's closing paragraph asked for "a rule that mines *semantic* reuse (fragments
computing the same function under different circuits) rather than the structural
reuse mined here".  This is that rule, and it is otherwise identical: R1
(fragments), R2's abstraction construction, the rewriting, R3's reuse gate and
R4's MDL score are all imported from `mine.py` unchanged.

The only change is the key under which occurrences are pooled.  Structurally
(§44, and `mine_multi`) two occurrences are the same abstraction exactly when
`Program.digest` agrees -- the right identity for a content-addressed *store*,
because it is what decides whether two modules share a file.  Here they are the
same abstraction when **(arity, truth table)** agrees.  Each pool elects one
representative circuit (fewest nodes, then digest) and every occurrence in the
pool is rewritten to a call to *that* module, which is semantics-preserving
because a fragment's truth table is taken in its own hole order.  The rewrite is
still executed against the task's rows before it is scored.

This is **not** a pre-registered arm and is not counted toward any falsification
criterion.  It is reported as an exploratory measurement of how much of this
track's negative is caused by the identity relation rather than by the corpus.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

from tcn.operators import Registry
from tcn.types import BOOL, Value

import mine
from mine_multi import MAX_HOLES, MAX_NODES, MIN_TASKS, _conforms


def truth_table(canon, cache):
    if canon.digest in cache:
        return cache[canon.digest]
    r = Registry()
    rows = []
    for bits in itertools.product((False, True), repeat=len(canon.inputs)):
        out, _ = canon.run({k: Value.of(BOOL, v) for (k, _), v in zip(canon.inputs, bits)},
                           registry=r)
        rows.append(bool(next(iter(out.values())).decoded))
    cache[canon.digest] = tuple(rows)
    return cache[canon.digest]


@dataclass
class Proposal:
    key: tuple
    digest: str
    canonical: object
    circuits: int
    nodes: int
    holes: int
    tasks: tuple
    entries: int
    occurrences: int
    saving_bits: int
    definition_bits: int
    baseline_bits: int
    rewritten_bits: int
    rank: int = 0

    def row(self):
        return {"truth_table": list(self.key[1]), "digest": self.digest,
                "circuits_pooled": self.circuits, "nodes": self.nodes,
                "holes": self.holes, "tasks": list(self.tasks), "entries": self.entries,
                "occurrences": self.occurrences, "saving_bits": self.saving_bits,
                "saving_per_entry": self.saving_bits / max(1, self.entries),
                "definition_bits": self.definition_bits,
                "baseline_bits": self.baseline_bits,
                "rewritten_bits": self.rewritten_bits, "rank": self.rank}


def propose(corpus, examples_by_entry, task_of, max_nodes=MAX_NODES,
            max_holes=MAX_HOLES, min_tasks=MIN_TASKS, verify=True):
    cache = {}
    pools = {}
    for entry, program in corpus.items():
        for root, S, canon, holes in mine.fragments(program, max_nodes, max_holes):
            key = (len(canon.inputs), truth_table(canon, cache))
            slot = pools.setdefault(key, {"reps": {}, "per_entry": {}})
            slot["reps"][canon.digest] = canon
            slot["per_entry"].setdefault(entry, []).append(
                {"root": root, "nodes": set(S), "holes": holes})

    baseline = {e: p.description_bits() for e, p in corpus.items()}
    total_baseline = sum(baseline.values())

    proposals = []
    for key, slot in pools.items():
        tasks = tuple(sorted({task_of[e] for e in slot["per_entry"]}))
        if len(tasks) < min_tasks:
            continue
        canon = min(slot["reps"].values(), key=lambda c: (len(c.nodes), c.digest))
        registry = Registry()
        name = registry.register_module(canon)
        if name != "module:" + canon.digest:
            continue
        rewritten_total, occurrences, hit_entries, ok = 0, 0, 0, True
        for entry, program in corpus.items():
            occs = slot["per_entry"].get(entry)
            if not occs:
                rewritten_total += baseline[entry]
                continue
            chosen = mine.disjoint_occurrences(sorted(occs, key=lambda o: o["root"]))
            try:
                rw = mine.rewrite(program, chosen, name, registry)
            except (TypeError, ValueError, KeyError):
                ok = False
                break
            if verify and not _conforms(rw, examples_by_entry[entry], registry):
                ok = False
                break
            rewritten_total += rw.description_bits()
            occurrences += len(chosen)
            hit_entries += 1
        if not ok or occurrences < min_tasks:
            continue
        definition = canon.description_bits()
        proposals.append(Proposal(
            key=key, digest=canon.digest, canonical=canon,
            circuits=len(slot["reps"]), nodes=len(canon.nodes),
            holes=len(canon.inputs), tasks=tasks, entries=hit_entries,
            occurrences=occurrences,
            saving_bits=total_baseline - rewritten_total - definition,
            definition_bits=definition, baseline_bits=total_baseline,
            rewritten_bits=rewritten_total))

    proposals.sort(key=lambda p: (-p.saving_bits, -len(p.tasks), -p.occurrences,
                                  -p.nodes, p.digest))
    for i, p in enumerate(proposals):
        p.rank = i + 1
    return proposals
