"""``quantification`` — generalized quantifiers.

Compositional semantics and logical language.
"""

from __future__ import annotations

import random

from .._structure import Ident, Pred
from ..lesson import Lesson
from ..generators.base import COLORS, _scene, _scene_term


def gen_quantification(rng: random.Random, ctx):
    """The truth value is chosen FIRST and the scene built to match it.

    Sampling a scene and reading off the truth value makes most quantified
    statements false, so a constant "no" scores ~0.73 and the lesson measures
    nothing. The verifier caught exactly this.
    """
    n = rng.randint(*ctx.span((3, 6), (5, 10)))
    color = rng.choice(COLORS)
    quant = rng.choice(["all", "some", "none", "exactly_two"])
    truth = rng.random() < 0.5
    target = {"all": n if truth else rng.randint(0, n - 1),
              "some": rng.randint(1, n) if truth else 0,
              "none": 0 if truth else rng.randint(1, n),
              "exactly_two": 2 if truth else rng.choice([c for c in range(0, n + 1) if c != 2])}[quant]
    others = [c for c in COLORS if c != color]
    objs = _scene(rng, n)
    for i, o in enumerate(objs):
        o["color"] = color if i < target else rng.choice(others)
    rng.shuffle(objs)
    for i, o in enumerate(objs):
        o["id"] = f"o{i}"
    k = sum(1 for o in objs if o["color"] == color)
    assert {"all": k == n, "some": k >= 1, "none": k == 0, "exactly_two": k == 2}[quant] == truth
    obs = _scene_term(objs, Pred("quant", Ident(quant), Ident(color)))
    return obs, ["yes", "no"], ("yes" if truth else "no"), {"count": k, "quantifier": quant}


class Quantification(Lesson):
    """Generalized quantifiers."""

    id = "quantification"
    level = 12
    tags = ("compositional-semantics", "logic")
    teaches = "generalized quantifiers"
    capabilities = ()
    axes = {'reasoning_depth': 3, 'world_complexity': 2}
    answers = ['yes', 'no']

    generate = staticmethod(gen_quantification)
