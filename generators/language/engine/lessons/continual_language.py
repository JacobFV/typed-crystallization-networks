"""``continual_language`` — lexical drift tracked across generations.

Analogy, causality, planning, and programs.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec, Term
from ..lesson import Lesson
from ..generators.base import COLORS
from ..generators.social import _nonce, _shuffled


def gen_continual_language(rng: random.Random, ctx):
    """A lexicon edited generation by generation; report a late meaning.

    Two edit types drive the drift: a *rename* keeps a meaning and changes its
    term, a *swap* keeps the terms and exchanges their meanings. Both are
    announced, so the final lexicon is a fold over the edit list — and the query
    is always chosen so that reading the generation-0 lexicon gives the wrong
    answer, which is what "old knowledge remains partially useful" has to mean if
    it is to be measured rather than asserted.
    """
    for _ in range(200):
        meanings = rng.sample(COLORS, 4)
        terms = [_nonce(rng, 3) for _ in range(4)]
        if len(set(terms)) < 4:
            continue
        state = dict(zip(terms, meanings))
        origin = dict(state)
        used = set(terms)
        events: list[Term] = []
        generations = rng.randint(*ctx.span((3, 4), (8, 11)))
        swaps = 0
        for g in range(generations):
            for _ in range(rng.randint(1, 2)):
                if rng.random() < 0.5:
                    old = rng.choice(list(state))
                    new = _nonce(rng, 3)
                    if new in used:
                        continue
                    used.add(new)
                    state[new] = state.pop(old)
                    events.append(Pred("rename", Num(g), Ident(old), Ident(new)))
                else:
                    t1, t2 = rng.sample(list(state), 2)
                    state[t1], state[t2] = state[t2], state[t1]
                    events.append(Pred("swap", Num(g), Ident(t1), Ident(t2)))
                    swaps += 1
        if swaps == 0:
            continue
        drifted = [t for t in state if origin.get(t) != state[t]]
        if not drifted:
            continue
        term = rng.choice(drifted)
        obs = Rec(generation_zero=Lst(_shuffled(rng, [Pred("means", Num(0), Ident(t), Ident(m))
                                                      for t, m in origin.items()])),
                  edits=Lst(events),
                  query=Pred("means", Num(generations), Ident(term)))
        return (obs, _shuffled(rng, meanings), state[term],
                {"generations": generations, "edits": len(events), "term": term,
                 "final_lexicon": {t: m for t, m in state.items()},
                 "initial_lexicon": dict(origin)})
    raise RuntimeError("continual_language: no admissible world")


class ContinualLanguage(Lesson):
    """Lexical drift tracked across generations."""

    id = "continual_language"
    level = 57
    tags = ("analogy", "causality", "planning", "programs")
    teaches = "lexical drift tracked across generations"
    capabilities = ('ontology_learning', 'variable_binding')
    axes = {'lexical_novelty': 4, 'discourse_horizon': 3, 'reasoning_depth': 3}

    generate = staticmethod(gen_continual_language)
