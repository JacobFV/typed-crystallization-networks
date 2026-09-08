"""``scientific_model_induction`` — select the model that fits, then predict.

Scientific induction and model discovery.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.science import _random_law, _shuffled


def gen_scientific_model_induction(rng: random.Random, ctx):
    """Four candidate models, one dataset, one held-out prediction.

    Exactly one model reproduces every observation, and all four disagree at the
    query point — so the answer vocabulary *is* the set of rival predictions and
    picking the right number is exactly picking the right theory. Model
    selection and prediction are graded as one act, which is what makes this
    induction rather than curve-fitting.
    """
    k = ctx.at(4, 6, default=4)                      # rival theories to tell apart
    for _ in range(300):
        laws = [_random_law(rng) for _ in range(k)]
        xs = rng.sample(range(-6, 7), 4)
        pool = [v for v in range(-8, 9) if v not in xs]
        xq = rng.choice(pool)
        true_i = rng.randrange(k)
        f_true = laws[true_i][1]
        data = [(x, f_true(x)) for x in xs]
        if [i for i, (_, f) in enumerate(laws) if all(f(x) == y for x, y in data)] != [true_i]:
            continue                                     # not exactly one survivor
        preds = [f(xq) for _, f in laws]
        if len(set(preds)) != k:
            continue                                     # the query does not discriminate
        order = _shuffled(rng, range(k))
        shown = Lst([Pred("theory", Ident(f"t{j + 1}"), laws[i][0]) for j, i in enumerate(order)])
        obs = Rec(data=Lst([Pred("observed", Num(x), Num(y)) for x, y in data]),
                  theories=shown, query=Pred("predict", Num(xq)))
        hidden = {"true_theory": f"t{order.index(true_i) + 1}", "x_query": xq,
                  "law": str(laws[true_i][0]), "predictions": preds}
        return obs, _shuffled(rng, preds), preds[true_i], hidden
    raise RuntimeError("scientific_model_induction: no admissible episode")


class ScientificModelInduction(Lesson):
    """Select the model that fits, then predict."""

    id = "scientific_model_induction"
    level = 69
    tags = ("science", "induction", "model-discovery")
    teaches = "select the model that fits, then predict"
    capabilities = ('scientific_induction', 'abstraction')
    axes = {'reasoning_depth': 4, 'compositional_depth': 3, 'world_complexity': 2}

    generate = staticmethod(gen_scientific_model_induction)
