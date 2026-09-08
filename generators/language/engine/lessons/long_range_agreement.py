"""Supplementary lesson: ``long_range_agreement`` — agreement with the head across attractors.

Supplementary syntax and semantics.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Rec, Tok
from ..lesson import Lesson
from ..generators.extra import (
    _shuffled, agree_forms, determiner, noun_forms, prepositions,
)


def gen_long_range_agreement(rng: random.Random, ctx):
    """Subject-verb agreement across a chain of attractor nouns of varying
    length: the classic diagnostic for whether a learner tracks the *head* of a
    phrase or merely the nearest noun. Attractor number is drawn independently,
    so the nearest-noun heuristic is wrong about half the time."""
    nouns, the = noun_forms(), determiner()
    n_att = rng.randint(*ctx.span((1, 4), (3, 8)))
    head_plural = rng.random() < 0.5
    head_sg, head_pl = rng.choice(nouns)
    toks = [the, head_pl if head_plural else head_sg]
    attractors = []
    for _ in range(n_att):
        sg, pl = rng.choice(nouns)
        att_plural = rng.random() < 0.5
        attractors.append((pl if att_plural else sg, att_plural))
        toks += [rng.choice(prepositions()), the, pl if att_plural else sg]
    toks.append("__")
    sg_form, pl_form = rng.choice(agree_forms())
    answer = pl_form if head_plural else sg_form
    obs = Rec(sentence=Lst([Tok(w) for w in toks if w]),
              query=Ident("main_verb_form"))
    return (obs, _shuffled(rng, [sg_form, pl_form]), answer,
            {"attractors": n_att, "distance": len(toks) - 2, "head_plural": head_plural,
             "attractor_numbers": [bool(p) for _, p in attractors]})


class LongRangeAgreement(Lesson):
    """Agreement with the head across attractors."""

    id = "long_range_agreement"
    level = 20
    tags = ("syntax", "semantics", "supplementary")
    teaches = "agreement with the head across attractors"
    capabilities = ()
    axes = {'grammar_complexity': 4, 'recursion_depth': 3, 'discourse_horizon': 3}

    generate = staticmethod(gen_long_range_agreement)
