"""``goal_revision`` — the goal new evidence makes infeasible.

Values and goal cognition.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.selfmodel import _labels, _rules, _shuffled


def gen_goal_revision(rng: random.Random, ctx):
    """New evidence removes a resource; which goal is no longer achievable?

    Availability propagates down a short derivation chain, so the depleted
    resource is usually not the one any goal names. Exactly one goal loses a
    requirement once the chain is followed.
    """
    n_res = ctx.at(6, 10, default=6)
    res = _labels(rng, "res", n_res)
    chain_len = rng.choice(ctx.among([[2, 3], [3, 4], [4, 5]]))
    chain = rng.sample(range(n_res), chain_len)   # chain[-1] is depleted at the base
    derived = [(chain[i], chain[i + 1]) for i in range(chain_len - 1)]
    unavailable = set(chain)
    free = [i for i in range(n_res) if i not in unavailable]
    assert len(free) >= 2

    gids = _labels(rng, "goal", 4)
    dead = rng.randrange(4)
    reqs: list[list[int]] = []
    for i in range(4):
        if i == dead:
            reqs.append(_shuffled(rng, [rng.choice(sorted(unavailable)), rng.choice(free)]))
        else:
            reqs.append(rng.sample(free, 2))
    assert sum(1 for r in reqs if any(x in unavailable for x in r)) == 1

    facts = [Pred("requires", Ident(gids[i]), Ident(res[r])) for i in range(4) for r in reqs[i]]
    facts += [Pred("derived_from", Ident(res[a]), Ident(res[b])) for a, b in derived]
    obs = Rec(goals=Lst(_shuffled(rng, facts)),
              evidence=Lst([Pred("depleted", Ident(res[chain[-1]]))]),
              rules=_rules("derived_from_a_b_means_a_is_available_only_while_b_is_available",
                           "a_depleted_resource_is_unavailable",
                           "a_goal_is_infeasible_iff_it_requires_an_unavailable_resource"),
              query=Ident("which_goal_to_drop"))
    return (obs, _shuffled(rng, gids), gids[dead],
            {"chain": [res[c] for c in chain], "unavailable": sorted(res[c] for c in chain)})


class GoalRevision(Lesson):
    """The goal new evidence makes infeasible."""

    id = "goal_revision"
    level = 160
    tags = ("values", "goals")
    teaches = "the goal new evidence makes infeasible"
    capabilities = ('value_learning', 'planning', 'causal_reasoning')
    axes = {'reasoning_depth': 4, 'world_complexity': 3, 'compositional_depth': 3}

    generate = staticmethod(gen_goal_revision)
