"""``multi_objective_reasoning`` — which option the Pareto front excludes.

Values and goal cognition.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.selfmodel import _labels, _rules, _shuffled


def gen_multi_objective_reasoning(rng: random.Random, ctx):
    """Which option is dominated — worse on nothing, better on nothing.

    Three objectives, all "higher is better", and exactly one option that some
    other option beats or matches on every axis. Scalarizing would answer a
    different question; the answer here is a fact about the Pareto front.
    """
    n_obj = ctx.at(3, 4, default=3)
    n_opt = ctx.at(4, 8, default=4)
    objectives = rng.sample(["throughput", "accuracy", "coverage", "robustness"], n_obj)
    for _ in range(200):
        vals = [[rng.randint(1, 9) for _ in range(n_obj)] for _ in range(n_opt)]

        def dominated(i: int) -> bool:
            return any(j != i and all(vals[j][k] >= vals[i][k] for k in range(n_obj))
                       and any(vals[j][k] > vals[i][k] for k in range(n_obj)) for j in range(n_opt))

        bad = [i for i in range(n_opt) if dominated(i)]
        if len(bad) == 1:
            break
    ids = _labels(rng, "option", n_opt)
    facts = [Pred("scores", Ident(ids[i]), Ident(objectives[k]), Num(vals[i][k]))
             for i in range(n_opt) for k in range(n_obj)]
    obs = Rec(objectives=Lst([Pred("maximize", Ident(o)) for o in _shuffled(rng, objectives)]),
              options=Lst(_shuffled(rng, facts)),
              rules=_rules("option_a_dominates_option_b_iff_a_is_at_least_as_good_on_every_objective_and_better_on_one",
                           "exactly_one_option_is_dominated"),
              query=Ident("dominated_option"))
    return (obs, _shuffled(rng, ids), ids[bad[0]],
            {"values": {ids[i]: vals[i] for i in range(n_opt)}, "objectives": objectives})


class MultiObjectiveReasoning(Lesson):
    """Which option the Pareto front excludes."""

    id = "multi_objective_reasoning"
    level = 158
    tags = ("values", "goals")
    teaches = "which option the Pareto front excludes"
    capabilities = ('value_learning', 'planning')
    axes = {'reasoning_depth': 4, 'world_complexity': 3, 'compositional_depth': 3}

    generate = staticmethod(gen_multi_objective_reasoning)
