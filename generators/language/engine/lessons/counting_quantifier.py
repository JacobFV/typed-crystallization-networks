"""Supplementary lesson: ``counting_quantifier`` — cardinal and proportional quantifiers.

Supplementary syntax and semantics.
"""

from __future__ import annotations

import random

from .._structure import Ident, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.base import COLORS
from ..generators.extra import _obj_list, _objects, _yesno


def gen_counting_quantifier(rng: random.Random, ctx):
    """Cardinal quantifiers over a scene: ``exactly k``, ``at least k``, and the
    proportional ``more X than Y``. The threshold ``k`` is chosen to make the
    target truth value hold, then the truth is *recomputed* from the scene."""
    n = rng.randint(*ctx.span((4, 7), (6, 12)))
    objs = _objects(rng, n)
    counts = {c: sum(1 for o in objs if o["color"] == c) for c in COLORS}
    want = rng.random() < 0.5
    kind = rng.choice(["exactly", "at_least", "more_than"])

    if kind == "exactly":
        color = rng.choice(COLORS)
        c = counts[color]
        k = c if want else rng.choice([j for j in range(0, n + 1) if j != c])
        truth = counts[color] == k
        query = Pred("exactly", Ident(color), Num(k))
        hidden = {"kind": kind, "color": color, "k": k, "count": c}
    elif kind == "at_least":
        present = [c for c in COLORS if counts[c] >= 1]
        color = rng.choice(present) if want and present else rng.choice(COLORS)
        k = rng.randint(1, counts[color]) if (want and counts[color] >= 1) else counts[color] + rng.randint(1, 2)
        truth = counts[color] >= k
        query = Pred("at_least", Ident(color), Num(k))
        hidden = {"kind": kind, "color": color, "k": k, "count": counts[color]}
    else:
        pairs = [(a, b) for a in COLORS for b in COLORS if a != b]
        pool = [p for p in pairs if (counts[p[0]] > counts[p[1]]) == want] or pairs
        a, b = rng.choice(pool)
        truth = counts[a] > counts[b]
        query = Pred("more_than", Ident(a), Ident(b))
        hidden = {"kind": kind, "colors": [a, b], "k": counts[a] - counts[b],
                  "count": counts[a]}

    hidden["n_objects"] = n
    answers, answer = _yesno(rng, truth)
    return Rec(scene=_obj_list(objs), query=query), answers, answer, hidden


class CountingQuantifier(Lesson):
    """Cardinal and proportional quantifiers."""

    id = "counting_quantifier"
    level = 19
    tags = ("syntax", "semantics", "supplementary")
    teaches = "cardinal and proportional quantifiers"
    capabilities = ()
    axes = {'reasoning_depth': 3, 'world_complexity': 3, 'recursion_depth': 2}
    answers = ['yes', 'no']

    generate = staticmethod(gen_counting_quantifier)
