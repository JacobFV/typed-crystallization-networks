"""Supplementary lesson: ``nesting_depth_compare`` — comparing two recursive structures.

Supplementary syntax and semantics.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Rec, Tok
from ..lesson import Lesson
from ..generators.extra import _max_depth, _nest, _shuffled


def gen_nesting_depth_compare(rng: random.Random, ctx):
    """Which of two bracket strings nests deeper? Both strings are padded with
    flat pairs so *length* carries no signal and only depth does."""
    d1, d2 = rng.sample(range(1, 6), 2)
    left_deep = rng.random() < 0.5
    ld, rd = (max(d1, d2), min(d1, d2)) if left_deep else (min(d1, d2), max(d1, d2))
    left = _nest(rng, ld) + "()" * rng.randint(*ctx.span((0, 4), (2, 8)))
    right = _nest(rng, rd) + "()" * rng.randint(*ctx.span((0, 4), (2, 8)))
    dl, dr = _max_depth(left), _max_depth(right)
    obs = Rec(left=Lst([Tok(c) for c in left]), right=Lst([Tok(c) for c in right]),
              query=Ident("deeper"))
    answer = "left" if dl > dr else "right"
    return (obs, _shuffled(rng, ["left", "right"]), answer,
            {"left_depth": dl, "right_depth": dr, "depth_gap": abs(dl - dr)})


class NestingDepthCompare(Lesson):
    """Comparing two recursive structures."""

    id = "nesting_depth_compare"
    level = 17
    tags = ("syntax", "semantics", "supplementary")
    teaches = "comparing two recursive structures"
    capabilities = ()
    axes = {'recursion_depth': 4, 'grammar_complexity': 3, 'reasoning_depth': 2}
    answers = ['left', 'right']

    generate = staticmethod(gen_nesting_depth_compare)
