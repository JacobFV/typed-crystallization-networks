"""Supplementary lesson: ``set_operations`` — boolean algebra over property sets.

Supplementary syntax and semantics.
"""

from __future__ import annotations

import random

from .._structure import Ident, Pred, Rec
from ..lesson import Lesson
from ..generators.base import COLORS, SHAPES
from ..generators.extra import _formula, _obj_list, _objects, _shuffled


def gen_set_operations(rng: random.Random, ctx):
    """Conjunction, disjunction and negation over property sets. Formulas are
    sampled and then *rejected* unless they denote exactly one object, so the
    description is a definite one and the answer is unique."""
    nc, ns = ctx.at(2, 3, default=2), ctx.at(2, 3, default=2)   # the colour x shape grid
    objs = _objects(rng, nc * ns)
    # colours and shapes each repeat, so no single property is a definite
    # description and only a boolean combination picks out one object
    cols = rng.sample(COLORS, nc)
    shapes = rng.sample(SHAPES, ns)
    for o, (c, s) in zip(objs, [(c, s) for c in cols for s in shapes]):
        o["color"], o["shape"] = c, s
    shown = _shuffled(rng, objs)
    for _ in range(80):
        sym, denoted, form = _formula(rng, objs)
        if len(denoted) == 1:
            break
    else:                                        # pragma: no cover - construction
        t = rng.choice(objs)
        sym = Pred("and", Pred("color", Ident(t["color"])), Pred("shape", Ident(t["shape"])))
        denoted, form = [t["id"]], "and"
    obs = Rec(scene=_obj_list(shown), query=Pred("select", sym))
    return (obs, _shuffled(rng, [o["id"] for o in objs]), denoted[0],
            {"form": form, "formula": str(sym), "denotes": denoted[0]})


class SetOperations(Lesson):
    """Boolean algebra over property sets."""

    id = "set_operations"
    level = 26
    tags = ("syntax", "semantics", "supplementary")
    teaches = "boolean algebra over property sets"
    capabilities = ()
    axes = {'compositional_depth': 4, 'reasoning_depth': 3, 'world_complexity': 2}

    generate = staticmethod(gen_set_operations)
