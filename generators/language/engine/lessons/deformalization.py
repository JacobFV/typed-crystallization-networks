"""``deformalization`` — formal structure to the reading that says it.

Open-ended epistemology.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec, Str
from ..lesson import Lesson
from ..generators.selfmodel import (_claim_pool, _claim_term, _formal_claim, _labels,
                                   _render_claim, _rules, _shuffled)


def gen_deformalization(rng: random.Random, ctx):
    """Formal structure in, the reading that says the same thing out.

    The mirror image of formalization: the same realizer generates all four
    candidate sentences from four distinct formal claims, so exactly one of them
    is the faithful rendering and the other three are systematic mis-readings.
    """
    true, picks = _claim_pool(rng)
    cands = [true] + picks
    n_cands = ctx.at(4, 8, default=4)
    while len(cands) < n_cands:                  # further mis-readings to sift through
        alt, more = _claim_pool(rng)
        for c in [alt] + more:
            if c not in cands and len(cands) < n_cands:
                cands.append(c)
    ids = _labels(rng, "gloss", n_cands)
    facts = [Pred("candidate", Ident(ids[i]), _claim_term(*cands[i])) for i in range(n_cands)]
    obs = Rec(theory=_formal_claim(*true),
              candidates=Lst(_shuffled(rng, facts)),
              rules=_rules("forall_x_implies_p_x_q_x_says_every_p_is_q",
                           "forall_x_implies_p_x_not_q_x_says_no_p_is_q",
                           "exists_x_and_p_x_q_x_says_some_p_is_q",
                           "exists_x_and_p_x_not_q_x_says_some_p_is_not_q"),
              query=Ident("which_gloss"))
    return (obs, _shuffled(rng, ids), ids[0],
            {"claim": list(true), "sentence": _render_claim(*true)})


class Deformalization(Lesson):
    """Formal structure to the reading that says it."""

    id = "deformalization"
    level = 154
    tags = ("open-ended-epistemology",)
    teaches = "formal structure to the reading that says it"
    capabilities = ('quantification', 'abstraction')
    axes = {'grammar_complexity': 3, 'compositional_depth': 3, 'reasoning_depth': 3}

    generate = staticmethod(gen_deformalization)
