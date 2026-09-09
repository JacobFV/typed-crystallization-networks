"""Supplementary lesson: ``nesting_depth_compare`` — comparing two recursive structures.

Supplementary syntax and semantics.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Rec, Tok
from ..lesson import Lesson
from ..generators.base import _dyck
from ..generators.extra import _max_depth, _nest, _shuffled


def gen_nesting_depth_compare(rng: random.Random, ctx):
    """Which of two bracket strings nests deeper? Both strings are padded with
    flat pairs so *length* carries no signal and only depth does."""
    d1, d2 = rng.sample(range(1, 6), 2)
    left_deep = rng.random() < 0.5
    ld, rd = (max(d1, d2), min(d1, d2)) if left_deep else (min(d1, d2), max(d1, d2))
    left = _nest(rng, ld) + "()" * rng.randint(*ctx.span((0, 4), (2, 8)))
    right = _nest(rng, rd) + "()" * rng.randint(*ctx.span((0, 4), (2, 8)))
    if ctx.hardens("nesting_depth_compare"):
        # Independent padding does not equalize anything: the deeper string is
        # still the longer one on average, and "where does the word *right*
        # start, as a fraction of the prompt" recovers that at 0.81.  Appending
        # the padding also leaves the deep region at the front, so a single byte
        # at a fixed offset reports the left string's depth.  Both sides are
        # redrawn with the *same* number of pairs and the flat pairs scattered
        # through the nesting, which makes the two strings identical in length
        # and in bracket counts with no fixed offset carrying the depth.
        pairs = rng.randint(*ctx.span((max(ld, rd) + 2, max(ld, rd) + 6),
                                      (max(ld, rd) + 4, max(ld, rd) + 12)))
        left, right = _dyck(rng, ld, pairs), _dyck(rng, rd, pairs)
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
