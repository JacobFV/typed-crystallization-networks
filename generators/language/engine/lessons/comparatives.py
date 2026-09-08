"""Supplementary lesson: ``comparatives`` — orderings over numeric attributes.

Supplementary syntax and semantics.
"""

from __future__ import annotations

import random

from .._structure import Ident, Pred, Rec
from ..lesson import Lesson
from ..generators.extra import _items, _objects, _shuffled


def gen_comparatives(rng: random.Random, ctx):
    """Orderings read off numeric attributes: bigger / smaller of a pair, the
    superlative over the scene, and proximity to an anchor. Sizes and positions
    are sampled without replacement so no comparison is ever a tie."""
    n = rng.randint(*ctx.span((3, 4), (4, 6)))
    objs = _objects(rng, n)
    for o, size, x in zip(objs, rng.sample(range(1, 20), n), rng.sample(range(0, 20), n)):
        o["size"], o["x"] = size, x
    kind = rng.choice(["bigger", "smaller", "largest", "smallest", "nearer"])

    if kind in ("bigger", "smaller"):
        a, b = rng.sample(objs, 2)
        pick = max if kind == "bigger" else min
        target = pick([a, b], key=lambda o: o["size"])
        query = Pred(kind, Ident(a["id"]), Ident(b["id"]))
        answers = _shuffled(rng, [a["id"], b["id"]])
        hidden = {"kind": kind, "sizes": {a["id"]: a["size"], b["id"]: b["size"]}}
    elif kind in ("largest", "smallest"):
        pick = max if kind == "largest" else min
        target = pick(objs, key=lambda o: o["size"])
        query = Ident(kind)
        answers = _shuffled(rng, [o["id"] for o in objs])
        hidden = {"kind": kind, "sizes": {o["id"]: o["size"] for o in objs}}
    else:
        anchor, a, b = rng.sample(objs, 3)
        if abs(a["x"] - anchor["x"]) == abs(b["x"] - anchor["x"]):   # never a tie
            target = max([a, b], key=lambda o: o["size"])
            query = Pred("bigger", Ident(a["id"]), Ident(b["id"]))
            kind = "bigger"
            hidden = {"kind": kind, "sizes": {a["id"]: a["size"], b["id"]: b["size"]}}
        else:
            target = min([a, b], key=lambda o: abs(o["x"] - anchor["x"]))
            query = Pred("nearer", Ident(a["id"]), Ident(b["id"]), Ident(anchor["id"]))
            hidden = {"kind": kind, "anchor": anchor["id"],
                      "xs": {o["id"]: o["x"] for o in (anchor, a, b)}}
        answers = _shuffled(rng, [a["id"], b["id"]])

    return Rec(scene=_items(objs), query=query), answers, target["id"], hidden


class Comparatives(Lesson):
    """Orderings over numeric attributes."""

    id = "comparatives"
    level = 25
    tags = ("syntax", "semantics", "supplementary")
    teaches = "orderings over numeric attributes"
    capabilities = ()
    axes = {'reasoning_depth': 3, 'world_complexity': 3, 'compositional_depth': 2}

    generate = staticmethod(gen_comparatives)
