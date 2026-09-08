"""``language_design`` — choose the language that expresses a task set in fewest symbols.

Reflective computation and language design.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.reflective import _coin_cost, _labels, _nonces, _shuffled


def gen_language_design(rng: random.Random, ctx):
    """Four invented languages, one task set: which says it in fewest symbols?

    Each language is a lexicon of numerals plus one compositional rule
    (juxtaposition sums values), so the cost of expressing a meaning is an
    exact minimum-coin computation and the cost of the task set is a sum. The
    generator rejects any world without a strict argmin, so "design" here is a
    decidable comparison rather than a matter of taste.
    """
    fallback = None
    for _ in range(300):
        ids = _labels(rng, "L", 4)
        sets = []
        for _ in range(4):
            extra = rng.sample([2, 3, 4, 5, 6, 7, 8, 9], rng.randint(1, 3))
            sets.append(sorted({1, *extra}))
        tasks = rng.sample(range(6, 22), ctx.at(3, 9, default=3))
        costs = {i: sum(_coin_cost(t, s) for t in tasks) for i, s in zip(ids, sets)}
        best = min(costs.values())
        winners = [i for i in ids if costs[i] == best]
        cand = (ids, sets, tasks, costs, winners[0])
        if fallback is None:
            fallback = cand
        if len(winners) == 1:
            fallback = cand
            break
    ids, sets, tasks, costs, answer = fallback
    langs = []
    for lid, s in zip(ids, sets):
        words = _nonces(rng, len(s), 3)
        langs.append(Pred("language", Ident(lid),
                          Lst([Pred("word", Ident(w), Num(v)) for w, v in zip(words, s)])))
    obs = Rec(languages=Lst(_shuffled(rng, langs)),
              composition=Pred("rule", Ident("juxtaposition"), Pred("sums_word_values")),
              tasks=Lst([Num(t) for t in tasks]),
              query=Ident("fewest_symbols_for_all_tasks"))
    return obs, _shuffled(rng, ids), answer, {"costs": costs, "tasks": tasks,
                                              "lexicons": {k: v for k, v in zip(ids, sets)}}


class LanguageDesign(Lesson):
    """Choose the language that expresses a task set in fewest symbols."""

    id = "language_design"
    level = 117
    tags = ("reflective-computation", "language-design")
    teaches = "choose the language that expresses a task set in fewest symbols"
    capabilities = ('abstraction', 'metareasoning', 'open_ended_discovery')
    axes = {'compositional_depth': 4, 'reasoning_depth': 4, 'lexical_novelty': 4}

    generate = staticmethod(gen_language_design)
