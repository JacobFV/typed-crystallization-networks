"""``resource_bounded_reasoning`` — the best operator you can actually afford.

Problem formulation and hierarchical agency.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.epistemics import OPERATORS, _shuffled


def gen_resource_bounded_reasoning(rng: random.Random, ctx):
    """Reasoning costs something; spend the budget where it pays.

    Every operator states a cost, a success probability and a payoff, and the
    budget rules some of them out — including, most of the time, the one with
    the highest expected value overall, so "pick the best operator" and "pick
    the best affordable operator" come apart. Expected values are compared in
    scaled integers, so there are no floating-point ties.
    """
    n_ops = ctx.at(5, 7, default=5)              # operators to compare
    for _ in range(400):
        names = rng.sample(OPERATORS, n_ops)
        budget = rng.randint(4, 9)
        ops = {}
        for nm in names:
            ops[nm] = {"cost": rng.randint(1, 12), "p": rng.randrange(10, 100, 5),
                       "payoff": rng.randint(4, 30)}
        ev = {nm: ops[nm]["p"] * ops[nm]["payoff"] - 100 * ops[nm]["cost"] for nm in names}
        afford = [nm for nm in names if ops[nm]["cost"] <= budget]
        if len(afford) < 2 or len(afford) == len(names):
            continue
        best = max(afford, key=lambda nm: ev[nm])
        if sum(1 for nm in afford if ev[nm] == ev[best]) != 1:
            continue
        overall = max(names, key=lambda nm: ev[nm])
        if sum(1 for nm in names if ev[nm] == ev[overall]) != 1:
            continue
        if (overall != best) == (rng.random() < 0.6):
            break
    else:                                     # pragma: no cover - construction
        pass

    obs = Rec(operators=Lst(_shuffled(rng, [
                  Pred("operator", Ident(nm), Pred("cost", Num(ops[nm]["cost"])),
                       Pred("success_percent", Num(ops[nm]["p"])),
                       Pred("payoff", Num(ops[nm]["payoff"]))) for nm in names])),
              budget=Num(budget),
              query=Ident("best_affordable_operator"))
    return (obs, _shuffled(rng, names), best,
            {"budget": budget, "answer": best, "unconstrained_best": overall,
             "expected_values": {nm: ev[nm] / 100 for nm in names}})


class ResourceBoundedReasoning(Lesson):
    """The best operator you can actually afford."""

    id = "resource_bounded_reasoning"
    level = 108
    tags = ("problem-formulation", "hierarchical-agency")
    teaches = "the best operator you can actually afford"
    capabilities = ('metareasoning', 'decision_theory', 'computational_cost')
    axes = {'reasoning_depth': 4, 'computational_budget': 4, 'uncertainty': 3}

    generate = staticmethod(gen_resource_bounded_reasoning)
