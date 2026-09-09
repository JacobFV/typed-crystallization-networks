"""``presupposition`` — asserted vs presupposed content under negation.

Language as action.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.base import COLORS, NAMES, SHAPES
from ..generators.semantics import ACTIVITIES, _shuffled


def gen_presupposition(rng: random.Random, ctx):
    """Asserted vs presupposed vs denied vs unrelated, under a polarity switch.

    Negation is the diagnostic: negating the utterance flips the asserted
    content to ``denied`` but leaves the presupposition standing, so a learner
    that treats the whole utterance as one proposition cannot separate the two
    layers. The label is drawn first and the polarity chosen to realize it.
    """
    # With no other utterance in the episode the polarity word is a global
    # feature of the whole prompt, and reading it together with the query's
    # predicate name classifies 0.868 of episodes without locating anything.
    n_extra = (ctx.at(1, 5, default=1) if ctx.hardens("presupposition")
               else ctx.at(0, 5, default=0))
    label = rng.choice(["asserted", "presupposed", "denied", "neither"])
    if label == "asserted":
        polarity = "affirm"
    elif label == "denied":
        polarity = "negate"
    else:
        polarity = rng.choice(["affirm", "negate"])

    family = rng.choice(["again", "definite"])
    if family == "again":
        subj, other_subj = rng.sample(NAMES, 2)
        act, other_act = rng.sample(ACTIVITIES, 2)
        utter = Pred("utterance", Ident(polarity), Ident("again"), Ident(subj), Ident(act))
        assertion = ("does", subj, act)
        presup = ("did_before", subj, act)
        unrelated = (rng.choice(["does", "did_before"]), other_subj, other_act)
        trigger = "again"
        key, okey = (subj, act), (other_subj, other_act)
        pool = [(x, y) for x in NAMES for y in ACTIVITIES
                if (x, y) not in {(subj, act), (other_subj, other_act)}]
    else:
        color, other_color = rng.sample(COLORS, 2)
        shape, other_shape = rng.sample(SHAPES, 2)
        utter = Pred("utterance", Ident(polarity), Ident("the_x_is_on_the_table"),
                     Ident(color), Ident(shape))
        assertion = ("on_table", color, shape)
        presup = ("exists", color, shape)
        unrelated = (rng.choice(["on_table", "exists"]), other_color, other_shape)
        trigger = "the_x_is_on_the_table"
        key, okey = (color, shape), (other_color, other_shape)
        pool = [(x, y) for x in COLORS for y in SHAPES
                if (x, y) not in {(color, shape), (other_color, other_shape)}]

    if ctx.hardens("presupposition"):
        # A `neither` proposition sharing *neither* argument with the utterance
        # is separable by raw character overlap alone. One shared argument makes
        # the overlap the same as an asserted proposition's, so the queried pair
        # has to actually be matched against the utterance.
        mixed = (key[0], okey[1]) if rng.random() < 0.5 else (okey[0], key[1])
        unrelated = (unrelated[0], mixed[0], mixed[1])
        pool = [p for p in pool if p != mixed]

    # utterances about wholly different pairs: they carry layers of their own,
    # so the queried proposition has to be traced back to the utterance that
    # licenses it before its layer can be read off the polarity
    others = [Pred("utterance", Ident(rng.choice(["affirm", "negate"])),
                   Ident(trigger), Ident(x), Ident(y))
              for x, y in (rng.sample(pool, n_extra) if n_extra else ())]
    candidate = {"asserted": assertion, "denied": assertion,
                 "presupposed": presup, "neither": unrelated}[label]
    extra_fields = {"context": Lst(others)} if others else {}
    obs = Rec(said=utter, **extra_fields,
              layers=Lst([Ident("asserted"), Ident("presupposed"), Ident("denied"),
                          Ident("neither")]),
              query=Pred("status_of", Ident(candidate[0]), Ident(candidate[1]),
                         Ident(candidate[2])))
    vocab = _shuffled(rng, ["asserted", "presupposed", "denied", "neither"])
    return obs, vocab, label, {"family": family, "polarity": polarity,
                               "candidate": list(candidate)}


class Presupposition(Lesson):
    """Asserted vs presupposed content under negation."""

    id = "presupposition"
    level = 25
    tags = ("pragmatics", "language-as-action")
    teaches = "asserted vs presupposed content under negation"
    capabilities = ('belief_modeling', 'proof_search')
    axes = {'reasoning_depth': 3, 'compositional_depth': 3, 'ambiguity': 2}
    answers = ['asserted', 'presupposed', 'denied', 'neither']

    generate = staticmethod(gen_presupposition)
