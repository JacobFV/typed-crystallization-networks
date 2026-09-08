"""Supplementary lesson: ``negation`` — negation and its scope over scene properties.

Supplementary syntax and semantics.
"""

from __future__ import annotations

import random

from .._structure import Ident, Pred, Rec
from ..lesson import Lesson
from ..generators.base import COLORS, SHAPES
from ..generators.extra import _obj_list, _objects, _yesno


def gen_negation(rng: random.Random, ctx):
    """Explicit negation over scene properties, in three scopes: negation of a
    property of a named object, negation inside an existential, and negation of
    the existential itself. The truth value is balanced by choosing the query's
    arguments, then recomputed from the scene."""
    n = rng.randint(*ctx.span((3, 5), (5, 10)))
    objs = _objects(rng, n)
    want = rng.random() < 0.5
    kind = rng.choice(["not_color", "exists_not", "none"])

    if kind == "not_color":
        o = rng.choice(objs)
        color = o["color"] if not want else rng.choice([c for c in COLORS if c != o["color"]])
        truth = o["color"] != color
        query = Pred("holds", Pred("not", Pred("color", Ident(o["id"]), Ident(color))))
        hidden = {"kind": kind, "object": o["id"], "color": color}
    elif kind == "exists_not":
        options = [(s, c) for s in SHAPES for c in COLORS
                   if any(o["shape"] == s and o["color"] != c for o in objs) == want]
        shape, color = rng.choice(options) if options else (rng.choice(SHAPES), rng.choice(COLORS))
        truth = any(o["shape"] == shape and o["color"] != color for o in objs)
        query = Pred("exists", Ident(shape), Pred("not", Ident(color)))
        hidden = {"kind": kind, "shape": shape, "color": color}
    else:
        present = {o["color"] for o in objs}
        pool = [c for c in COLORS if (c not in present) == want] or COLORS
        color = rng.choice(pool)
        truth = all(o["color"] != color for o in objs)
        query = Pred("none", Ident(color))
        hidden = {"kind": kind, "color": color}

    answers, answer = _yesno(rng, truth)
    return Rec(scene=_obj_list(objs), query=query), answers, answer, hidden


class Negation(Lesson):
    """Negation and its scope over scene properties."""

    id = "negation"
    level = 24
    tags = ("syntax", "semantics", "supplementary")
    teaches = "negation and its scope over scene properties"
    capabilities = ()
    axes = {'reasoning_depth': 3, 'compositional_depth': 3, 'world_complexity': 2}
    answers = ['yes', 'no']

    generate = staticmethod(gen_negation)
