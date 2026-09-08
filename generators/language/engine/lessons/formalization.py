"""``formalization`` — informal claim to formal structure.

Open-ended epistemology.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec, Str
from ..lesson import Lesson
from ..generators.selfmodel import (_claim_pool, _claim_term, _formal_claim, _labels,
                                   _render_claim, _rules, _shuffled)


def gen_formalization(rng: random.Random, ctx):
    """Informal claim in, formal structure out.

    The formal claim is drawn first and rendered into English by a fixed
    realizer, so the intended reading is exact rather than a matter of taste. The
    distractors differ from it by one decision each — the quantifier, the
    polarity, the direction of the predication, or the predicate itself — so
    every one of them is a sentence the learner must be able to tell apart.
    """
    n_cands = ctx.at(4, 8, default=4)
    true, picks = _claim_pool(rng)
    cands = [true] + picks
    for _ in range(40):                          # more near-misses to tell apart
        if len(cands) >= n_cands:
            break
        _other, more = _claim_pool(rng)
        for c in more:
            if c not in cands and len(cands) < n_cands:
                cands.append(c)
    sentence = _render_claim(*true)
    ids = _labels(rng, "form", len(cands))
    facts = [Pred("candidate", Ident(ids[i]), _formal_claim(*cands[i]))
             for i in range(len(cands))]
    obs = Rec(statement=_claim_term(*true),
              candidates=Lst(_shuffled(rng, facts)),
              rules=_rules("forall_x_implies_p_x_q_x_says_every_p_is_q",
                           "forall_x_implies_p_x_not_q_x_says_no_p_is_q",
                           "exists_x_and_p_x_q_x_says_some_p_is_q",
                           "exists_x_and_p_x_not_q_x_says_some_p_is_not_q"),
              query=Ident("which_formalization"))
    return (obs, _shuffled(rng, ids), ids[0],
            {"claim": list(true), "sentence": sentence})


class Formalization(Lesson):
    """Informal claim to formal structure."""

    id = "formalization"
    level = 153
    tags = ("open-ended-epistemology",)
    teaches = "informal claim to formal structure"
    capabilities = ('quantification', 'abstraction', 'ontology_learning')
    axes = {'grammar_complexity': 3, 'compositional_depth': 3, 'reasoning_depth': 3}

    generate = staticmethod(gen_formalization)
