"""``belief_state`` — nested belief that can differ from the truth.

Language as action.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.base import NAMES
from ..generators.semantics import PLACES, _shuffled


def gen_belief_state(rng: random.Random, ctx):
    """First- and second-order belief over a moving object.

    An agent's belief is the last move it witnessed; what ``a`` believes ``b``
    believes is fixed by the moves ``a`` could have seen ``b`` see, i.e. the
    smaller of their two horizons. Both rules are stated symbolically in the
    observation so the target is exactly determined, and the belief queries
    disagree with the truth whenever an agent left early — which is most
    episodes.
    """
    locs = rng.sample(PLACES, ctx.at(4, 6, default=4))
    n_moves = rng.randint(*ctx.span((2, 3), (4, 5)))
    seq = rng.sample(locs, n_moves + 1)
    a, b = rng.sample(NAMES, 2)
    horizon = {a: rng.randint(0, n_moves), b: rng.randint(0, n_moves)}
    kind = rng.choice(["truth", "belief", "nested"])
    if kind == "truth":
        answer, query = seq[n_moves], Pred("where_is", Ident("key"))
    elif kind == "belief":
        who = rng.choice([a, b])
        answer, query = seq[horizon[who]], Pred("believes_where", Ident(who))
    else:
        x, y = rng.sample([a, b], 2)
        answer = seq[min(horizon[x], horizon[y])]
        query = Pred("believes_that_believes", Ident(x), Ident(y))

    obs = Rec(start=Pred("start", Ident("key"), Ident(seq[0])),
              moves=Lst([Pred("move", Num(i), Ident("key"), Ident(seq[i]))
                         for i in range(1, n_moves + 1)]),
              witnesses=Lst(_shuffled(rng, [Pred("witnessed_upto", Ident(g), Num(h))
                                            for g, h in horizon.items()])),
              rules=Lst([Pred("rule", Ident("believes"), Pred("last_move_witnessed")),
                         Pred("rule", Ident("believes_that_believes"),
                              Pred("min_of_both_horizons"))]),
              query=query)
    return obs, _shuffled(rng, locs), answer, {"kind": kind, "sequence": seq,
                                               "horizons": {k: v for k, v in horizon.items()},
                                               "truth": seq[n_moves]}


class BeliefState(Lesson):
    """Nested belief that can differ from the truth."""

    id = "belief_state"
    level = 35
    tags = ("pragmatics", "language-as-action")
    teaches = "nested belief that can differ from the truth"
    capabilities = ('belief_modeling', 'multi_agent_coordination')
    axes = {'reasoning_depth': 4, 'discourse_horizon': 3, 'world_complexity': 3}

    generate = staticmethod(gen_belief_state)
