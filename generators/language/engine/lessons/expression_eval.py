"""Supplementary lesson: ``expression_eval`` — compositional evaluation of a nested expression.

Supplementary syntax and semantics.
"""

from __future__ import annotations

import random

from .._structure import Ident, Pred, Rec
from ..lesson import Lesson
from ..generators.extra import _OPS, _expr, _shuffled


def gen_expression_eval(rng: random.Random, ctx):
    """Evaluate a nested arithmetic expression. The value is computed alongside
    the tree, so the label is exact by construction rather than by a checker."""
    depth = rng.randint(*ctx.span((1, 3), (2, 4)))
    for _ in range(40):
        op = rng.choice(_OPS)                    # the root is always an operator
        left, lv = _expr(rng, depth - 1)
        right, rv = _expr(rng, depth - 1)
        val = {"+": lv + rv, "-": lv - rv, "*": lv * rv}[op]
        expr = Pred(op, left, right)
        if abs(val) <= 40:
            break
    offsets = _shuffled(rng, [d for d in range(-6, 7) if d != 0])
    distractors: list[int] = []
    for d in offsets:
        if val + d not in distractors:
            distractors.append(val + d)
        if len(distractors) == 4:
            break
    return (Rec(expression=expr, query=Ident("value")),
            _shuffled(rng, [val] + distractors), val,
            {"depth": depth, "value": val, "expression": str(expr)})


class ExpressionEval(Lesson):
    """Compositional evaluation of a nested expression."""

    id = "expression_eval"
    level = 18
    tags = ("syntax", "semantics", "supplementary")
    teaches = "compositional evaluation of a nested expression"
    capabilities = ()
    axes = {'recursion_depth': 4, 'compositional_depth': 4, 'reasoning_depth': 3}

    generate = staticmethod(gen_expression_eval)
