"""``invariance_discovery`` — the transformation a property survives.

Scientific induction and model discovery.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.science import _labels, _property, _shuffled, _transform


def gen_invariance_discovery(rng: random.Random, ctx):
    """Which transformation leaves the named property alone?

    A structure, a property stated symbolically, and four generated
    transformations — permutation, translation, scaling, sign flip, reversal,
    local edit. Exactly one preserves the property, established by *applying all
    four and comparing*, and the episode is rejected otherwise, so "invariant"
    never means "probably invariant". The pairing of property and transformation
    is resampled every episode, so no transformation is a safe bet.
    """
    n = ctx.at(5, 12, default=5)
    for _ in range(400):
        v = [rng.randint(-6, 9) for _ in range(n)]
        prop = rng.choice([Pred("sum"), Pred("spread"), Pred("multiset"),
                           Pred("parity_of_sum"), Pred("at", Num(rng.randrange(n))),
                           Pred("count_positive"), Pred("maximum")])
        perm = _shuffled(rng, range(n))
        pool = [Pred("translate", Num(rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]))),
                Pred("scale", Num(rng.choice([-2, -1, 2, 3]))),
                Pred("negate"), Pred("reverse"), Pred("sort"),
                Pred("permute", Lst([Num(p) for p in perm])),
                Pred("replace", Num(rng.randrange(n)), Num(rng.randint(-6, 9)))]
        cands = _shuffled(rng, pool)[:4]
        base = _property(prop, v)
        keep = [i for i, tspec in enumerate(cands) if _property(prop, _transform(tspec, v)) == base]
        if len(keep) != 1:
            continue
        labels = _labels("f", 4)
        answer = labels[keep[0]]
        obs = Rec(structure=Lst([Num(x) for x in v]),
                  transformations=Lst([Pred("transformation", Ident(labels[j]), c)
                                       for j, c in enumerate(cands)]),
                  query=Pred("preserves", prop))
        hidden = {"property": str(prop), "invariant": str(cands[keep[0]]), "answer": answer,
                  "structure": list(v)}
        return obs, _shuffled(rng, labels), answer, hidden
    raise RuntimeError("invariance_discovery: no admissible episode")


class InvarianceDiscovery(Lesson):
    """The transformation a property survives."""

    id = "invariance_discovery"
    level = 77
    tags = ("science", "induction", "model-discovery")
    teaches = "the transformation a property survives"
    capabilities = ('abstraction', 'scientific_induction')
    axes = {'reasoning_depth': 4, 'compositional_depth': 3, 'world_complexity': 2}

    generate = staticmethod(gen_invariance_discovery)
