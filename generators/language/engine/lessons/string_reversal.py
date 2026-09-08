"""Supplementary lesson: ``string_reversal`` — indexing into an implied reversed structure.

Supplementary syntax and semantics.
"""

from __future__ import annotations

import random

from .._structure import Lst, Num, Pred, Rec, Tok
from ..lesson import Lesson
from ..generators.extra import _shuffled


def gen_string_reversal(rng: random.Random, ctx):
    """The k-th symbol of the reversal: indexing into a structure that is never
    written down, only implied."""
    alpha = list("abcdef")
    n = rng.randint(*ctx.span((3, 9), (7, 15)))
    s = [rng.choice(alpha) for _ in range(n)]
    k = rng.randrange(n)
    obs = Rec(symbols=Lst([Tok(c) for c in s]), query=Pred("reversed_at", Num(k)))
    return obs, _shuffled(rng, alpha), s[n - 1 - k], {"length": n, "index": k, "string": "".join(s)}


class StringReversal(Lesson):
    """Indexing into an implied reversed structure."""

    id = "string_reversal"
    level = 16
    tags = ("syntax", "semantics", "supplementary")
    teaches = "indexing into an implied reversed structure"
    capabilities = ()
    axes = {'recursion_depth': 3, 'discourse_horizon': 2, 'compositional_depth': 2}

    generate = staticmethod(gen_string_reversal)
