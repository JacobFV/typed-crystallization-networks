"""``open_ended_question_generation`` — ask the question that eliminates the most.

Open-ended epistemology.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.selfmodel import _labels, _rules, _shuffled


def gen_open_ended_question_generation(rng: random.Random, ctx):
    """Which question is worth asking: the one that eliminates the most hypotheses.

    Every live hypothesis states what it predicts for every candidate question, so
    a question's worst-case yield is the number of hypotheses ruled out whichever
    way the answer comes back. This is expected-information-gain reduced to
    counting, which is what makes "the agent decides what to ask" scoreable.
    """
    n_h, n_q = ctx.at(8, 18, default=8), ctx.at(4, 6, default=4)
    for _ in range(200):
        table = [[rng.random() < 0.5 for _ in range(n_q)] for _ in range(n_h)]
        vals = []
        for q in range(n_q):
            yes = sum(1 for h in range(n_h) if table[h][q])
            vals.append(n_h - max(yes, n_h - yes))
        top = max(vals)
        if vals.count(top) == 1 and top > 0:
            break
    best = vals.index(top)

    hids = _labels(rng, "hyp", n_h)
    qids = _labels(rng, "question", n_q)
    facts = [Pred("predicts", Ident(hids[h]), Ident(qids[q]),
                  Ident("yes" if table[h][q] else "no"))
             for h in range(n_h) for q in range(n_q)]
    obs = Rec(hypotheses=Lst([Pred("live", Ident(h)) for h in _shuffled(rng, hids)]),
              predictions=Lst(_shuffled(rng, facts)),
              rules=_rules("asking_a_question_eliminates_every_hypothesis_whose_prediction_differs_from_the_answer",
                           "the_value_of_a_question_is_the_number_eliminated_in_the_worse_case",
                           "choose_the_question_of_greatest_value"),
              query=Ident("most_informative_question"))
    return (obs, _shuffled(rng, qids), qids[best],
            {"values": {qids[q]: vals[q] for q in range(n_q)}, "n_hypotheses": n_h})


class OpenEndedQuestionGeneration(Lesson):
    """Ask the question that eliminates the most."""

    id = "open_ended_question_generation"
    level = 147
    tags = ("open-ended-epistemology",)
    teaches = "ask the question that eliminates the most"
    capabilities = ('open_ended_discovery', 'scientific_induction', 'metareasoning')
    axes = {'reasoning_depth': 4, 'world_complexity': 4, 'ambiguity': 3}

    generate = staticmethod(gen_open_ended_question_generation)
