"""``unification`` — structural symbolic matching.

Symbols, grounding, and elementary language.
"""

from __future__ import annotations

import random
import string

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.base import NAMES
from ..generators.extra import _shuffled


def gen_unification(rng: random.Random, ctx):
    """Prolog-style term matching: parent(X,bob) vs parent(alice,bob) → X=alice."""
    arity = ctx.at(2, 5, default=2)
    args = rng.sample(NAMES, arity)
    var = rng.choice(list(string.ascii_uppercase[:5]))
    pos = rng.randrange(arity)
    pattern = Pred("parent", *[Ident(var) if i == pos else Ident(x) for i, x in enumerate(args)])
    fact = Pred("parent", *[Ident(x) for x in args])
    if not ctx.hardens("unification"):
        obs = Rec(pattern=pattern, fact=fact, query=Pred("unify", Ident(var)))
        return obs, NAMES, args[pos], {"var": var, "position": pos}
    # With a single fact the binding is the one name that occurs exactly once:
    # every other name is printed twice, in the pattern and in the fact.  A
    # perceptron over "how often does this option occur" reaches 1.000 that way.
    # Near-miss facts disagree with the pattern in exactly one *bound* position,
    # so they never unify, and the occurrence counts stop separating anything.
    others = [Pred("parent", *(Ident(x) for x in _near_miss(rng, args, pos, arity)))
              for _ in range(rng.randint(2, 3))]
    facts = _shuffled(rng, [fact] + others)
    obs = Rec(pattern=pattern, facts=Lst(facts), query=Pred("unify", Ident(var)))
    return (obs, NAMES, args[pos],
            {"var": var, "position": pos, "facts": len(facts)})


def _near_miss(rng: random.Random, args, pos: int, arity: int):
    """A fact that agrees with the pattern everywhere but one bound position."""
    cand = list(args)
    j = rng.choice([i for i in range(arity) if i != pos])
    cand[j] = rng.choice([n for n in NAMES if n != args[j]])
    cand[pos] = rng.choice(list(NAMES))
    return cand


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
