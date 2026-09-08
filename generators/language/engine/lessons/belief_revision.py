"""``belief_revision`` — the minimal retraction that restores consistency.

Mathematics and formal reasoning.
"""

from __future__ import annotations

import random
from typing import Any

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.mathematics import _fsym, _label_items, _rand_formula, _sat, _shuffled


def gen_belief_revision(rng: random.Random, ctx):
    """One new sentence makes the knowledge base inconsistent; retract the least.

    The conflict is built so that exactly one existing sentence has to go:
    dropping it restores satisfiability, dropping any other leaves the core
    intact. Consistency is decided by enumerating every valuation, and the
    generator verifies the whole retraction profile (one repair, four
    non-repairs) before the episode is emitted."""
    atoms = list("pqrst")
    n_rest = ctx.at(4, 7, default=4)              # innocent beliefs beside the culprit
    budget = ctx.at(60, 240, default=60)
    for _ in range(400):
        x, y = rng.sample(atoms, 2)
        ax, ay = ("atom", x), ("atom", y)
        template = rng.randrange(4)
        if template == 0:
            new, culprit = ("and", ax, ("not", ay)), ("imp", ax, ay)
        elif template == 1:
            new, culprit = ("not", ax), ax
        elif template == 2:
            new, culprit = ("imp", ax, ay), ("and", ax, ("not", ay))
        else:
            new, culprit = ("or", ax, ay), ("and", ("not", ax), ("not", ay))
        rest: list[Any] = []
        for _ in range(budget):
            if len(rest) == n_rest:
                break
            f = _rand_formula(rng, atoms, 2)
            if f in rest or f == culprit or f == new:
                continue
            if _sat(rest + [f, new], atoms) and _sat(rest + [f, culprit], atoms):
                rest.append(f)
        if len(rest) != n_rest:
            continue
        kb = [culprit] + rest
        if not _sat(kb, atoms) or _sat(kb + [new], atoms):
            continue
        profile = [_sat([s for s in kb if s is not t] + [new], atoms) for t in kb]
        if profile != [True] + [False] * n_rest:
            continue
        break
    else:                                                # pragma: no cover - construction
        kb = [("atom", "p"), ("atom", "q"), ("atom", "r"), ("atom", "s"), ("atom", "t")]
        new = ("not", ("atom", "p"))

    shown, label_of = _label_items(rng, kb, prefix="b")
    obs = Rec(beliefs=Lst([Pred("believes", Ident(lab), _fsym(f)) for lab, f in shown]),
              incoming=_fsym(new),
              query=Ident("minimal_retraction_restoring_consistency"))
    return obs, _shuffled(rng, [lab for lab, _ in shown]), label_of[0], {
        "culprit": str(_fsym(kb[0])), "incoming": str(_fsym(new))}


class BeliefRevision(Lesson):
    """The minimal retraction that restores consistency."""

    id = "belief_revision"
    level = 92
    tags = ("mathematics", "formal-reasoning")
    teaches = "the minimal retraction that restores consistency"
    capabilities = ('consistency', 'minimal_change', 'revision')
    axes = {'reasoning_depth': 5, 'compositional_depth': 3, 'world_complexity': 3}

    generate = staticmethod(gen_belief_revision)
