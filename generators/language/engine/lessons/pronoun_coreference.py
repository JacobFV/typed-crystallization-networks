"""``pronoun_coreference`` — resolving a pronoun to its one valid antecedent.

Language as action.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec, Tok
from ..lesson import Lesson
from ..generators.base import NAMES
from ..generators.extra import (
    ACTIVE_LANGUAGE, DEFAULT_LANGUAGE, _shuffled, adverbs, gender, intransitive,
    pronoun, supplies, then_word, verbs,
)


def _in_english(rng: random.Random, ctx):
    """The same episode, built from English material throughout.

    This lesson turns on the pronoun distinguishing its two antecedents, so a
    language with one genderless third person -- Finnish *hän*, Turkish *o*,
    Hungarian *ő* -- cannot present it at all. Those languages do supply the
    rest of the sentence, and taking that while falling back for the pronoun
    alone would put the question in English inside a Finnish clause.
    """
    token = ACTIVE_LANGUAGE.set(DEFAULT_LANGUAGE)
    try:
        return gen_pronoun_coreference(rng, ctx)
    finally:
        ACTIVE_LANGUAGE.reset(token)


def gen_pronoun_coreference(rng: random.Random, ctx):
    """Two entities, one pronoun, exactly one grammatically valid antecedent —
    the other entity is ruled out by gender, so the episode has a single correct
    reading. Filler material varies the distance to the antecedent."""
    # Before any draw, so the fallback sees the same random stream and picks
    # the same people: checking later re-ran the generator mid-stream and the
    # English version came out about a different pair.
    if not supplies("pronouns"):
        return _in_english(rng, ctx)
    genders = gender()
    fem = [n for n in NAMES if genders[n] == "f"]
    masc = [n for n in NAMES if genders[n] == "m"]
    she_ref, he_ref = rng.choice(fem), rng.choice(masc)
    first, second = (she_ref, he_ref) if rng.random() < 0.5 else (he_ref, she_ref)
    # The pronoun is the thing the lesson is about, so it has to be the
    # pronoun of the language the episode is read in. Choosing on the gender
    # keeps the random draw identical across languages and only the word
    # different, which is what the cross-language invariant requires.
    pronouns = pronoun()
    sex = rng.choice(["f", "m"])
    pron = pronouns[sex]
    referent = she_ref if sex == "f" else he_ref
    filler = [rng.choice(adverbs()) for _ in range(rng.randint(*ctx.span((0, 2), (5, 9))))]
    toks = ([first, rng.choice(verbs()), second] + filler
            + [then_word(), pron, rng.choice(intransitive())])
    obs = Rec(discourse=Lst([Tok(w) for w in toks]), query=Pred("refers_to", Ident(pron)))
    return (obs, _shuffled(rng, [she_ref, he_ref]), referent,
            {"pronoun": pron, "referent": referent,
             "distractor": he_ref if pron == "she" else she_ref,
             "distance": toks.index(pron) - toks.index(referent)})


class PronounCoreference(Lesson):
    """Resolving a pronoun to its one valid antecedent."""

    id = "pronoun_coreference"
    level = 23
    tags = ("pragmatics", "language-as-action")
    teaches = "resolving a pronoun to its one valid antecedent"
    capabilities = ()
    axes = {'discourse_horizon': 3, 'ambiguity': 2, 'compositional_depth': 2}

    generate = staticmethod(gen_pronoun_coreference)
