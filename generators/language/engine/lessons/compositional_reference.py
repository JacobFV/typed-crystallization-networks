"""``compositional_reference`` — recursive denotation.

Compositional semantics and logical language.
"""

from __future__ import annotations

import random

from .._structure import Ident, Pred
from ..lesson import Lesson
from ..generators.base import COLORS, _scene, _scene_term


def gen_compositional_reference(rng: random.Random, ctx):
    """'the red object left of the blue cube' — recursive denotation."""
    n_objs = ctx.at(4, 8, default=4)
    objs = _scene(rng, n_objs)
    # distinct x so 'left of' is unambiguous, but assigned by a shuffled
    # permutation: otherwise object *id order* correlates with the answer and a
    # constant-guessing agent scores far above chance.
    xs = [2 * i for i in range(n_objs)]
    rng.shuffle(xs)
    for o, x in zip(objs, xs):
        o["x"] = x
    anchor = max(objs, key=lambda o: o["x"]) if rng.random() < 0.5 else objs[rng.randrange(n_objs)]
    lefts = [o for o in objs if o["x"] < anchor["x"]]
    if not lefts:
        anchor = max(objs, key=lambda o: o["x"])
        lefts = [o for o in objs if o["x"] < anchor["x"]]
    tgt = rng.choice(lefts)
    tgt_color = tgt["color"]
    while sum(1 for o in lefts if o["color"] == tgt_color) > 1:
        tgt_color = rng.choice(COLORS)
        for o in lefts:
            o["color"] = rng.choice(COLORS)
        tgt = rng.choice(lefts)
        tgt_color = tgt["color"]
    q = Pred("the", Ident(tgt_color), Pred("left_of", Ident(anchor["color"]), Ident(anchor["shape"])))
    return _scene_term(objs, q), [o["id"] for o in objs], tgt["id"], {"anchor": anchor, "target": tgt}


class CompositionalReference(Lesson):
    """Recursive denotation."""

    id = "compositional_reference"
    level = 13
    tags = ("compositional-semantics", "logic")
    teaches = "recursive denotation"
    capabilities = ()
    axes = {'compositional_depth': 3, 'world_complexity': 2, 'ambiguity': 1}

    generate = staticmethod(gen_compositional_reference)
