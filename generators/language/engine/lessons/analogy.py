"""``analogy`` — a:b::c:? over a relational structure.

Analogy, causality, planning, and programs.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.base import NAMES
from ..generators.extra import _shuffled


def gen_analogy(rng: random.Random, ctx):
    """``a : b :: c : ?`` over a relational structure. Two functional relations
    are laid over the same six entities and disagree on every entity, so exactly
    one of them explains the pair ``a:b`` and the answer is that relation applied
    to ``c``. Which relation was used is never named."""
    nodes = list(NAMES)
    k = ctx.at(2, 3, default=2)                  # rival relations over the same entities
    for _ in range(200 if k == 2 else 20000):
        perms = [_shuffled(rng, nodes) for _ in range(k)]
        if all(all(p[i] != nodes[i] for i in range(len(nodes))) for p in perms) and \
                all(all(perms[j][i] != perms[l][i] for i in range(len(nodes)))
                    for j in range(k) for l in range(j + 1, k)):
            break
    else:                                        # pragma: no cover - derangement
        perms = [nodes[j:] + nodes[:j] for j in range(1, k + 1)]
    rels = rng.sample(["follows", "outranks", "precedes", "mirrors"], k)
    maps = {r: dict(zip(nodes, p)) for r, p in zip(rels, perms)}
    rel = rng.choice(rels)
    a = rng.choice(nodes)
    b = maps[rel][a]
    c = rng.choice([x for x in nodes if x not in (a, b)])
    d = maps[rel][c]
    edges = [Pred(name, Ident(x), Ident(m[x])) for name, m in maps.items() for x in nodes]
    obs = Rec(graph=Lst(_shuffled(rng, edges)),
              query=Pred("analogy", Ident(a), Ident(b), Ident(c)))
    return (obs, _shuffled(rng, nodes), d,
            {"relation": rel, "a": a, "b": b, "c": c, "answer": d})


class Analogy(Lesson):
    """A:b::c:? over a relational structure."""

    id = "analogy"
    level = 29
    tags = ("analogy", "causality", "planning", "programs")
    teaches = "a:b::c:? over a relational structure"
    capabilities = ()
    axes = {'reasoning_depth': 4, 'compositional_depth': 3, 'world_complexity': 3}

    generate = staticmethod(gen_analogy)
