"""``goal_generation`` — the intermediate objective the top goal now requires.

Values and goal cognition.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.selfmodel import _labels, _rules, _shuffled


def gen_goal_generation(rng: random.Random, ctx):
    """Which intermediate objective does the top goal now require?

    The top goal decomposes into subgoals with prerequisites among them; some are
    already achieved. Exactly one unachieved subgoal has all of its prerequisites
    in place, and that is the objective the agent should generate next.
    """
    n = ctx.at(6, 12, default=6)
    for _ in range(300):
        deps = {i: set() for i in range(n)}
        for i in range(1, n):
            for j in range(i):
                if rng.random() < 0.4:
                    deps[i].add(j)
        done: set[int] = set()
        for i in range(n):
            if deps[i] <= done and rng.random() < 0.55:
                done.add(i)
        ready = [i for i in range(n) if i not in done and deps[i] <= done]
        if len(ready) == 1 and len(done) < n:
            break
    ids = _labels(rng, "sub", n)
    top = "mission"
    facts = [Pred("part_of", Ident(ids[i]), Ident(top)) for i in range(n)]
    facts += [Pred("requires", Ident(ids[i]), Ident(ids[j])) for i in range(n) for j in deps[i]]
    facts += [Pred("achieved", Ident(ids[i])) for i in sorted(done)]
    obs = Rec(decomposition=Lst(_shuffled(rng, facts)),
              top_goal=Ident(top),
              rules=_rules("the_top_goal_needs_every_subgoal_that_is_part_of_it",
                           "a_subgoal_is_actionable_iff_it_is_unachieved_and_every_subgoal_it_requires_is_achieved",
                           "exactly_one_subgoal_is_actionable"),
              query=Ident("next_subgoal"))
    return (obs, _shuffled(rng, ids), ids[ready[0]],
            {"achieved": [ids[i] for i in sorted(done)], "actionable": ids[ready[0]]})


class GoalGeneration(Lesson):
    """The intermediate objective the top goal now requires."""

    id = "goal_generation"
    level = 161
    tags = ("values", "goals")
    teaches = "the intermediate objective the top goal now requires"
    capabilities = ('planning', 'value_learning')
    axes = {'reasoning_depth': 4, 'compositional_depth': 4, 'world_complexity': 3}

    generate = staticmethod(gen_goal_generation)
