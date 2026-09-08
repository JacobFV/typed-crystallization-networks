"""``dsl_invention`` — which invented DSL makes the recurring computations shortest.

Reflective computation and language design.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec, Tok
from ..lesson import Lesson
from ..generators.reflective import _dsl_cost, _labels, _nonces, _shuffled


def gen_dsl_invention(rng: random.Random, ctx):
    """Which invented DSL makes the recurring computations shortest?

    Every candidate offers two macros over the same primitives; writing a task
    is a shortest-segmentation problem, solved by dynamic programming. The tasks
    are built from repeated motifs, so a DSL that names the right motif wins —
    and which one that is cannot be read off the surface, only computed.
    """
    fallback = None
    for _ in range(300):
        prims = _nonces(rng, 3, 2)
        motifs = [tuple(rng.choice(prims) for _ in range(rng.randint(2, 3))) for _ in range(3)]
        tasks = []
        for _ in range(rng.randint(*ctx.span((2, 3), (6, 8)))):
            seq: list[str] = []
            while len(seq) < 6:
                seq += list(rng.choice(motifs)) if rng.random() < 0.75 else [rng.choice(prims)]
            tasks.append(seq[:9])
        ids = _labels(rng, "D", 4)
        kits = []
        for _ in range(4):
            pool = list(motifs) + [tuple(rng.choice(prims) for _ in range(rng.randint(2, 3)))]
            kits.append(rng.sample(pool, 2))
        costs = {i: sum(_dsl_cost(t, k) for t in tasks) for i, k in zip(ids, kits)}
        best = min(costs.values())
        winners = [i for i in ids if costs[i] == best]
        cand = (prims, tasks, ids, kits, costs, winners[0])
        if fallback is None:
            fallback = cand
        if len(winners) == 1:
            fallback = cand
            break
    prims, tasks, ids, kits, costs, answer = fallback
    dsls = []
    for did, kit in zip(ids, kits):
        names = _nonces(rng, len(kit), 3)
        dsls.append(Pred("dsl", Ident(did),
                         Lst([Pred("macro", Ident(nm), Lst([Tok(t) for t in body]))
                              for nm, body in zip(names, kit)])))
    obs = Rec(primitives=Lst([Ident(p) for p in prims]),
              tasks=Lst([Lst([Tok(t) for t in seq]) for seq in tasks]),
              dsls=Lst(_shuffled(rng, dsls)),
              cost_rule=Pred("cost", Pred("one_per_written_token")),
              query=Ident("fewest_tokens_for_all_tasks"))
    return obs, _shuffled(rng, ids), answer, {"costs": costs, "n_tasks": len(tasks)}


class DslInvention(Lesson):
    """Which invented DSL makes the recurring computations shortest."""

    id = "dsl_invention"
    level = 118
    tags = ("reflective-computation", "language-design")
    teaches = "which invented DSL makes the recurring computations shortest"
    capabilities = ('abstraction', 'program_synthesis', 'open_ended_discovery')
    axes = {'compositional_depth': 5, 'reasoning_depth': 4, 'lexical_novelty': 4}

    generate = staticmethod(gen_dsl_invention)
