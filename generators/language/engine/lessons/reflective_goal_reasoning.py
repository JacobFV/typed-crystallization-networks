"""``reflective_goal_reasoning`` — which goal to abandon given derived conflicts.

Values and goal cognition.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.selfmodel import _labels, _rules, _shuffled


def gen_reflective_goal_reasoning(rng: random.Random, ctx):
    """Reason about the goals themselves: which one has to go?

    Conflict is not declared, it is *derived* — two goals conflict when they
    demand different values of the same state variable. Exactly one such pair
    exists, and the goal to abandon is its lower-priority member, so both the
    conflict and the resolution have to be computed.
    """
    n_g, n_v = ctx.at(4, 8, default=4), ctx.at(4, 20, default=4)
    varz = _labels(rng, "var", n_v)
    gids = _labels(rng, "goal", n_g)
    for _ in range(400):
        reqs = []
        for _i in range(n_g):
            vs = rng.sample(range(n_v), 2)
            reqs.append({v: rng.randint(0, 1) for v in vs})
        clashes = [(i, j) for i in range(n_g) for j in range(i + 1, n_g)
                   if any(v in reqs[j] and reqs[j][v] != reqs[i][v] for v in reqs[i])]
        if len(clashes) == 1:
            break
    prio = list(rng.sample(range(1, n_g + 1), n_g))
    i, j = clashes[0]
    drop = i if prio[i] < prio[j] else j

    facts = [Pred("needs", Ident(gids[g]), Ident(varz[v]), Num(val))
             for g in range(n_g) for v, val in reqs[g].items()]
    facts += [Pred("priority", Ident(gids[g]), Num(prio[g])) for g in range(n_g)]
    obs = Rec(goals=Lst(_shuffled(rng, facts)),
              rules=_rules("two_goals_conflict_iff_they_need_different_values_of_the_same_variable",
                           "a_higher_priority_number_means_a_more_important_goal",
                           "resolve_a_conflict_by_dropping_the_lower_priority_goal_of_the_pair"),
              query=Ident("which_goal_to_abandon"))
    return (obs, _shuffled(rng, gids), gids[drop],
            {"conflict": [gids[i], gids[j]], "priorities": {gids[g]: prio[g] for g in range(n_g)}})


class ReflectiveGoalReasoning(Lesson):
    """Which goal to abandon given derived conflicts."""

    id = "reflective_goal_reasoning"
    level = 162
    tags = ("values", "goals")
    teaches = "which goal to abandon given derived conflicts"
    capabilities = ('metareasoning', 'value_learning', 'planning')
    axes = {'reasoning_depth': 5, 'compositional_depth': 4, 'ambiguity': 2}

    generate = staticmethod(gen_reflective_goal_reasoning)
