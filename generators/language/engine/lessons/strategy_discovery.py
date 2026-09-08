"""``strategy_discovery`` — inducing the invariant behind a problem family.

Problem formulation and hierarchical agency.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.epistemics import _game_draw, _game_pools, _shuffled


def gen_strategy_discovery(rng: random.Random, ctx):
    """A problem family with one reusable strategy behind every instance.

    The worked instances are positions in a take-away game with a per-episode
    move set; the answer to each is the winning move, or that there is none.
    Solving instances one at a time is search; the family rewards inducing the
    invariant (the losing positions), which is what makes the new position
    answerable at a glance. Positions are restricted to those with a unique
    winning move so the target is exact.
    """
    hi = ctx.at(22, 34, default=22)
    for _ in range(200):
        moves = sorted(rng.sample([1, 2, 3, 4, 5], rng.randint(*ctx.span((2, 3), (4, 5)))))
        naming = {m: f"take_{m}" for m in moves}
        pools = _game_pools(moves, hi, "none", naming)
        if len(pools) >= 3:                   # else one answer would dominate
            break
    shown = []
    used: list[int] = []
    for _ in range(4):
        k, lab = _game_draw(rng, pools, used)
        used.append(k)
        shown.append((k, lab))
    q, answer = _game_draw(rng, pools, used)
    vocab = [f"take_{m}" for m in moves] + ["none"]

    obs = Rec(rules=Lst([Pred("may_remove", Num(m)) for m in moves]
                        + [Pred("outcome", Pred("player_taking_last_token_wins"))]),
              solved_instances=Lst(_shuffled(rng, [Pred("instance", Num(k), Ident(a))
                                                   for k, a in shown])),
              query=Pred("winning_move_at", Num(q)))
    return (obs, _shuffled(rng, vocab), answer,
            {"moves": moves, "position": q, "answer": answer,
             "examples": {str(k): a for k, a in shown}})


class StrategyDiscovery(Lesson):
    """Inducing the invariant behind a problem family."""

    id = "strategy_discovery"
    level = 111
    tags = ("problem-formulation", "hierarchical-agency")
    teaches = "inducing the invariant behind a problem family"
    capabilities = ('strategy_induction', 'game_reasoning', 'abstraction')
    axes = {'reasoning_depth': 5, 'recursion_depth': 3, 'world_complexity': 3}

    generate = staticmethod(gen_strategy_discovery)
