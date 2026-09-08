"""``unification`` — structural symbolic matching.

Symbols, grounding, and elementary language.
"""

from __future__ import annotations

import random
import string

from .._structure import Ident, Pred, Rec
from ..lesson import Lesson
from ..generators.base import NAMES


def gen_unification(rng: random.Random, ctx):
    """Prolog-style term matching: parent(X,bob) vs parent(alice,bob) → X=alice."""
    arity = ctx.at(2, 5, default=2)
    args = rng.sample(NAMES, arity)
    var = rng.choice(list(string.ascii_uppercase[:5]))
    pos = rng.randrange(arity)
    pattern = Pred("parent", *[Ident(var) if i == pos else Ident(x) for i, x in enumerate(args)])
    fact = Pred("parent", *[Ident(x) for x in args])
    obs = Rec(pattern=pattern, fact=fact, query=Pred("unify", Ident(var)))
    return obs, NAMES, args[pos], {"var": var, "position": pos}


class Unification(Lesson):
    """Structural symbolic matching."""

    id = "unification"
    level = 11
    tags = ("symbols", "grounding", "elementary-language")
    teaches = "structural symbolic matching"
    capabilities = ()
    axes = {'compositional_depth': 3, 'reasoning_depth': 3}
    answers = ['alice', 'bob', 'carol', 'dave', 'erin', 'frank']

    generate = staticmethod(gen_unification)
