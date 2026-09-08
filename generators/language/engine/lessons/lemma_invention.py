"""``lemma_invention`` — the intermediate proposition that most shortens a proof.

Mathematics and formal reasoning.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.mathematics import _derivation_costs, _horn_theory, _label_items, _ladder_theory, _shuffled


def gen_lemma_invention(rng: random.Random, ctx):
    """Which intermediate proposition, taken as a lemma, buys the most?

    Proof cost is tree size, so a proposition that is re-derived inside several
    branches is worth far more as a lemma than one used once. The generator
    computes the goal's minimal derivation cost with and without each of four
    candidate lemmas and keeps the episode only when one candidate's saving is
    strictly the largest."""
    for att in range(400):
        if att < 380:
            atoms, facts, rules = _horn_theory(rng, n_atoms=ctx.at(8, 14, default=8))
            cands_all = None
        else:                                            # pragma: no cover - budget
            atoms, facts, rules = _ladder_theory(rng)
            cands_all = atoms[3:7]
        d0 = _derivation_costs(facts, rules, atoms)
        goal = atoms[-1]
        if d0[goal] is None or d0[goal] < 5:
            continue
        cands = cands_all or [a for a in atoms
                              if a not in facts and a != goal and d0[a] is not None]
        if len(cands) < 4:
            continue
        cands = list(cands) if cands_all else rng.sample(cands, 4)
        savings = []
        for c in cands:
            dc = _derivation_costs(list(facts) + [c], rules, atoms)
            savings.append(d0[goal] - (dc[goal] if dc[goal] is not None else d0[goal]))
        best = max(savings)
        if best <= 0 or savings.count(best) != 1:
            continue
        win = cands[savings.index(best)]
        order = [win] + [c for c in cands if c != win]
        break
    else:                                                # pragma: no cover - unreachable
        raise RuntimeError("lemma_invention: no well-posed episode found")

    shown, label_of = _label_items(rng, order, prefix="k")
    obs = Rec(axioms=Lst([Pred("fact", Ident(a)) for a in _shuffled(rng, facts)]),
              rules=Lst([Pred("rule", Ident(nm), Ident(h), Lst([Ident(b) for b in body]))
                         for nm, h, body in _shuffled(rng, rules)]),
              goal=Ident(goal),
              candidates=Lst([Pred("lemma", Ident(lab), Ident(a)) for lab, a in shown]),
              query=Ident("lemma_that_most_shortens_the_proof"))
    return obs, _shuffled(rng, [a for _, a in shown]), order[0], {
        "goal": goal, "base_cost": d0[goal], "savings": [int(s) for s in savings]}


class LemmaInvention(Lesson):
    """The intermediate proposition that most shortens a proof."""

    id = "lemma_invention"
    level = 84
    tags = ("mathematics", "formal-reasoning")
    teaches = "the intermediate proposition that most shortens a proof"
    capabilities = ('proof_search', 'abstraction', 'cost_reasoning')
    axes = {'reasoning_depth': 5, 'compositional_depth': 4, 'recursion_depth': 3}

    generate = staticmethod(gen_lemma_invention)
