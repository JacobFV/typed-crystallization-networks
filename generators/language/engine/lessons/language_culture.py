"""``language_culture`` — iterated learning through a transmission bottleneck.

Analogy, causality, planning, and programs.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec, Term
from ..lesson import Lesson
from ..generators.social import CONCEPTS, WORDS, _shuffled


def gen_language_culture(rng: random.Random, ctx):
    """Each generation learns from a sample of the last one and fills the gaps.

    This is the iterated-learning setup in miniature: a learner hears words for
    only some of the meanings and, by the stated regularization rule, extends the
    last word it heard to every meaning it did not. Repeated over generations the
    lexicon collapses onto fewer forms — and which form survives is a fact about
    the transmission chain, computable exactly by replaying it, not about any
    property of the words themselves.
    """
    for _ in range(200):
        meanings = rng.sample(CONCEPTS, 4)
        lex = dict(zip(meanings, rng.sample(WORDS, 4)))
        origin = dict(lex)
        generations = rng.randint(*ctx.span((2, 3), (6, 8)))
        heard_facts: list[Term] = []
        for g in range(1, generations + 1):
            order = _shuffled(rng, meanings)[:rng.randint(2, 3)]
            heard = [(m, lex[m]) for m in order]
            for i, (m, w) in enumerate(heard):
                heard_facts.append(Pred("heard", Num(g), Num(i), Ident(m), Ident(w)))
            fallback = heard[-1][1]
            table = dict(heard)
            lex = {m: table.get(m, fallback) for m in meanings}
        drifted = [m for m in meanings if lex[m] != origin[m]]
        if not drifted:
            continue
        target = rng.choice(drifted)
        obs = Rec(founder_lexicon=Lst(_shuffled(rng, [Pred("says", Ident(m), Ident(w))
                                                      for m, w in origin.items()])),
                  transmission=Lst(heard_facts),
                  rule=Lst([Pred("unheard_meaning_takes", Pred("last_heard_word"))]),
                  query=Pred("word_for", Num(generations), Ident(target)))
        return (obs, _shuffled(rng, list(origin.values())), lex[target],
                {"generations": generations, "meaning": target,
                 "founder_lexicon": dict(origin), "final_lexicon": dict(lex),
                 "distinct_words_left": len(set(lex.values()))})
    raise RuntimeError("language_culture: no admissible world")


class LanguageCulture(Lesson):
    """Iterated learning through a transmission bottleneck."""

    id = "language_culture"
    level = 58
    tags = ("analogy", "causality", "planning", "programs")
    teaches = "iterated learning through a transmission bottleneck"
    capabilities = ('ontology_learning', 'multi_agent_coordination')
    axes = {'lexical_novelty': 4, 'discourse_horizon': 4, 'reasoning_depth': 3}

    generate = staticmethod(gen_language_culture)
