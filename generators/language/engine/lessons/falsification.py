"""``falsification`` — the observation that would refute the law.

Scientific induction and model discovery.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.base import COLORS, SHAPES
from ..generators.science import _labels, _lin, _shuffled


def gen_falsification(rng: random.Random, ctx):
    """Which single observation would refute the stated law?

    Half the episodes state a universal (``every X-shaped thing is C``) and half
    an equation. Exactly one candidate contradicts it; the others are consistent,
    and among them is always at least one that is *irrelevant* rather than
    confirming — vacuous satisfaction and genuine confirmation both count as
    "not a refutation", which is the distinction the lesson is about.
    """
    n = ctx.at(4, 8, default=4)                  # candidate observations to sift
    labels = _labels("o", n)
    if rng.random() < 0.5:
        shape = rng.choice(SHAPES)
        color = rng.choice(COLORS)
        others_shape = [s for s in SHAPES if s != shape]
        others_color = [c for c in COLORS if c != color]
        refuter = (shape, rng.choice(others_color))
        cases = [(shape, color),                                   # confirming instance
                 (rng.choice(others_shape), rng.choice(others_color)),   # irrelevant
                 (rng.choice(others_shape), color)]                # irrelevant
        for _ in range(n - 4):                                     # further irrelevant ones
            cases.append((rng.choice(others_shape), rng.choice(others_color)))
        rows = [refuter] + cases
        order = _shuffled(rng, range(n))
        answer = labels[order.index(0)]
        obs = Rec(law=Pred("all_are", Ident(shape), Ident(color)),
                  candidates=Lst([Pred("candidate", Ident(labels[j]),
                                       Ident(rows[i][0]), Ident(rows[i][1]))
                                  for j, i in enumerate(order)]),
                  query=Ident("refutes_the_law"))
        return obs, _shuffled(rng, labels), answer, {"mode": "universal",
                                                     "law": [shape, color], "answer": answer}
    a = rng.choice([-3, -2, -1, 1, 2, 3])
    b = rng.randint(-6, 6)
    xs = rng.sample(range(-7, 8), n)
    off = rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
    rows = [(xs[0], a * xs[0] + b + off)] + [(x, a * x + b) for x in xs[1:]]
    order = _shuffled(rng, range(n))
    answer = labels[order.index(0)]
    obs = Rec(law=Pred("eq", Ident("y"), _lin(a, "x", b)),
              candidates=Lst([Pred("candidate", Ident(labels[j]),
                                   Num(rows[i][0]), Num(rows[i][1]))
                              for j, i in enumerate(order)]),
              query=Ident("refutes_the_law"))
    return obs, _shuffled(rng, labels), answer, {"mode": "equation", "law": [a, b],
                                                 "offset": off, "answer": answer}


class Falsification(Lesson):
    """The observation that would refute the law."""

    id = "falsification"
    level = 72
    tags = ("science", "induction", "model-discovery")
    teaches = "the observation that would refute the law"
    capabilities = ('scientific_induction', 'quantification')
    axes = {'reasoning_depth': 3, 'compositional_depth': 2, 'ambiguity': 1}

    generate = staticmethod(gen_falsification)
