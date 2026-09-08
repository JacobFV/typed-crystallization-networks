"""``symbol_discrimination`` — category boundaries.

Symbols, grounding, and elementary language.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson


def gen_symbol_discrimination(rng: random.Random, ctx):
    """A hidden category boundary between near neighbours."""
    top = ctx.at(10, 20, default=10)             # how far the scale runs
    boundary = rng.randint(*ctx.span((3, 7), (6, 14)))
    v = rng.randint(0, top)
    examples = [(k, "high" if k >= boundary else "low") for k in range(0, top + 1, 2) if k != v]
    rng.shuffle(examples)
    shown = examples[:ctx.at(5, 10, default=5)]
    obs = Rec(examples=Lst([Pred("ex", Num(k), Ident(lab)) for k, lab in shown]),
              query=Pred("classify", Num(v)))
    return obs, ["low", "high"], ("high" if v >= boundary else "low"), {"boundary": boundary}


class SymbolDiscrimination(Lesson):
    """Category boundaries."""

    id = "symbol_discrimination"
    level = 3
    tags = ("symbols", "grounding", "elementary-language")
    teaches = "category boundaries"
    capabilities = ()
    axes = {'ambiguity': 1, 'reasoning_depth': 1}
    answers = ['low', 'high']

    generate = staticmethod(gen_symbol_discrimination)
