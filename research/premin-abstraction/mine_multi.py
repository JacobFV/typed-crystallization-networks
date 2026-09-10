"""§44's selection rule, over a corpus that may hold several programs per task.

R1 (fragments), R2 (abstraction, canonical identity) and the rewriting are
imported unchanged from `research/earned-abstraction/mine.py`.  Exactly two
things change, and both are forced by the corpus holding more than one program
per task:

*   **R3 counts base tasks, not corpus entries.**  An abstraction is eligible
    when it occurs in at least `MIN_TASKS` distinct *earlier tasks*.  On a
    one-program-per-task corpus this is identical to §44's rule.

*   **R4 sums over corpus entries.**  `saving(F) = sum_e bits(P_e) -
    sum_e bits(rewrite_e(F)) - bits(F)` over every entry `e`.  The definition is
    still charged exactly once, so a larger corpus amortises it over more call
    sites.  Absolute bit figures are therefore only comparable *within* a
    corpus; the ranking within a corpus is unaffected by corpus size scaling,
    and `saving_per_entry` is reported so the reader can see the effect.

Verification is unchanged in substance: a rewrite that changes any output
disqualifies the abstraction.  It is checked against the task's own targets
rather than by re-executing the original, which is the same test because every
corpus program is conformant by construction and is verified in `tcn` before
it enters the corpus.
"""
from __future__ import annotations

from dataclasses import dataclass

from tcn.operators import Registry

import mine

MAX_NODES = mine.MAX_NODES
MAX_HOLES = mine.MAX_HOLES
MIN_TASKS = mine.MIN_TASKS


@dataclass
class Proposal:
    digest: str
    canonical: object
    nodes: int
    holes: int
    tasks: tuple
    entries: int
    occurrences: int
    saving_bits: int
    definition_bits: int
    baseline_bits: int
    rewritten_bits: int
    execution_cost: float
    rank: int = 0

    def row(self):
        return {"digest": self.digest, "nodes": self.nodes, "holes": self.holes,
                "tasks": list(self.tasks), "entries": self.entries,
                "occurrences": self.occurrences, "saving_bits": self.saving_bits,
                "saving_per_entry": self.saving_bits / max(1, self.entries),
                "definition_bits": self.definition_bits,
                "baseline_bits": self.baseline_bits,
                "rewritten_bits": self.rewritten_bits,
                "execution_cost": self.execution_cost, "rank": self.rank}


def _conforms(program, examples, registry):
    for ex in examples:
        try:
            out, _, _ = program.execute(ex["inputs"], registry=registry)
        except (ValueError, TypeError, KeyError, IndexError, OverflowError):
            return False
        got = next(iter(out.values()))
        want = ex["targets"]["out"]
        if got.to_dict() != want.to_dict():
            return False
    return True


def propose(corpus, examples_by_entry, task_of, max_nodes=MAX_NODES,
            max_holes=MAX_HOLES, min_tasks=MIN_TASKS, verify=True):
    """Rank every eligible abstraction over a corpus of solved programs.

    `corpus`            entry id -> frozen pruned `Program`
    `examples_by_entry` entry id -> the rows the rewrite is verified against
    `task_of`           entry id -> the base task the entry solves
    """
    by_digest = {}
    for entry, program in corpus.items():
        for root, S, canon, holes in mine.fragments(program, max_nodes, max_holes):
            slot = by_digest.setdefault(canon.digest, {"canonical": canon, "per_entry": {}})
            slot["per_entry"].setdefault(entry, []).append(
                {"root": root, "nodes": set(S), "holes": holes})

    baseline = {e: p.description_bits() for e, p in corpus.items()}
    total_baseline = sum(baseline.values())

    proposals = []
    for digest, slot in by_digest.items():
        canon = slot["canonical"]
        tasks = tuple(sorted({task_of[e] for e in slot["per_entry"]}))
        if len(tasks) < min_tasks:
            continue
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
            digest=digest, canonical=canon, nodes=len(canon.nodes),
            holes=len(canon.inputs), tasks=tasks, entries=hit_entries,
            occurrences=occurrences,
            saving_bits=total_baseline - rewritten_total - definition,
            definition_bits=definition, baseline_bits=total_baseline,
            rewritten_bits=rewritten_total,
            execution_cost=canon.execution_cost(registry)))

    proposals.sort(key=lambda p: (-p.saving_bits, -len(p.tasks), -p.occurrences,
                                  -p.nodes, p.digest))
    for i, p in enumerate(proposals):
        p.rank = i + 1
    return proposals
