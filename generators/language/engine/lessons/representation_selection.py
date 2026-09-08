"""``representation_selection`` — which encoding answers the query in fewest steps.

Ontology and representation.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.ontology import ENCODINGS, ITEMS, _selection_costs, _shuffled, _unique_argmin


def gen_representation_selection(rng: random.Random, ctx):
    """One latent chain, four encodings, one query: which encoding costs least?

    ``graph`` is a shuffled edge list (each edge is found by scanning), ``sequence``
    an ordered list (scan to the item, then walk), ``set`` an unordered bag with
    O(1) membership and no successor at all, ``table`` keyed rows that must be
    indexed first. The four coefficients of the cost model are resampled every
    episode and shown in the observation, so no encoding is a standing answer:
    the four costs have to be computed.
    """
    target = rng.choice(list(ENCODINGS))
    fallback = None
    for _ in range(400):
        n = rng.randint(*ctx.span((4, 7), (6, 8)))
        chain = rng.sample(list(ITEMS), n)
        edges = _shuffled(rng, [(chain[i], chain[i + 1]) for i in range(n - 1)])
        u, h, b, l = rng.choice([1, 2]), rng.choice([1, 2, 3]), rng.choice([1, 2]), rng.choice([1, 2, 3])
        if target == "set" or rng.random() < 0.45:
            k, x = 0, rng.choice(chain)
        else:
            i = rng.randrange(n - 1)
            k, x = rng.randint(1, min(3, n - 1 - i)), chain[i]
        costs = _selection_costs(chain, edges, u, h, b, l, x, k)
        win = _unique_argmin(costs)
        if win is None:
            continue
        fallback = (n, chain, edges, u, h, b, l, k, x, costs, win)
        if win == target:
            break
    if fallback is None:                             # pragma: no cover - construction
        raise RuntimeError("representation_selection: no episode")
    n, chain, edges, u, h, b, l, k, x, costs, win = fallback

    rules = [
        Pred("cost", Ident("sequence"), Pred("scan_per_item"), Num(u)),
        Pred("cost", Ident("sequence"), Pred("per_step"), Num(h)),
        Pred("cost", Ident("graph"), Pred("scan_per_edge"), Num(u)),
        Pred("cost", Ident("set"), Ident("lookup"), Num(l)),
        Pred("cost", Ident("set"), Pred("per_step"), Ident("unsupported")),
        Pred("cost", Ident("table"), Pred("index_per_row"), Num(b)),
        Pred("cost", Ident("table"), Ident("lookup"), Num(l)),
    ]
    obs = Rec(
        graph=Lst([Pred("edge", Ident(a), Ident(c)) for a, c in edges]),
        sequence=Lst([Ident(c) for c in chain]),
        set=Lst([Ident(c) for c in _shuffled(rng, chain)]),
        table=Lst([Pred("row", Ident(a), Ident(c)) for a, c in _shuffled(rng, edges)]),
        cost_model=Lst(rules),
        query=Pred("cheapest_encoding", Ident(x), Num(k)),
    )
    hidden = {"costs": {kk: vv for kk, vv in costs.items()}, "k": k, "n": n,
              "coefficients": [u, h, b, l], "answer": win}
    return obs, _shuffled(rng, ENCODINGS), win, hidden


class RepresentationSelection(Lesson):
    """Which encoding answers the query in fewest steps."""

    id = "representation_selection"
    level = 64
    tags = ("ontology", "representation")
    teaches = "which encoding answers the query in fewest steps"
    capabilities = ('representation_choice', 'cost_model_reasoning', 'planning')
    axes = {'reasoning_depth': 4, 'world_complexity': 3, 'compositional_depth': 2}

    generate = staticmethod(gen_representation_selection)
