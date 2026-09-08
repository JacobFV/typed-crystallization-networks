"""``metareasoning`` — choosing how to reason, fallbacks included.

Problem formulation and hierarchical agency.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.epistemics import OPERATORS, _shuffled


def gen_metareasoning(rng: random.Random, ctx):
    """Choosing how to reason, with a fallback if the reasoning fails.

    Each operator can fail, and failure is not worth zero — there is a stated
    fallback payoff, which differs per operator. So the ranking under
    ``p x payoff - cost`` is deliberately made to disagree with the true ranking
    under ``p x payoff + (1-p) x fallback - cost`` in most episodes; the greedy
    reading of the same table is wrong.
    """
    for _ in range(400):
        names = rng.sample(OPERATORS, ctx.at(5, 7, default=5))     # OPERATORS has 7
        budget = rng.randint(6, 12)
        ops = {}
        for nm in names:
            ops[nm] = {"cost": rng.randint(1, budget), "p": rng.randrange(10, 100, 5),
                       "payoff": rng.randint(5, 30), "fallback": rng.randint(0, 20)}
        ev = {nm: ops[nm]["p"] * ops[nm]["payoff"] + (100 - ops[nm]["p"]) * ops[nm]["fallback"]
                  - 100 * ops[nm]["cost"] for nm in names}
        naive = {nm: ops[nm]["p"] * ops[nm]["payoff"] - 100 * ops[nm]["cost"] for nm in names}
        best = max(names, key=lambda nm: ev[nm])
        nbest = max(names, key=lambda nm: naive[nm])
        if sum(1 for nm in names if ev[nm] == ev[best]) != 1:
            continue
        if (best != nbest) == (rng.random() < 0.7):
            break
    else:                                     # pragma: no cover - construction
        pass

    obs = Rec(reasoning_operators=Lst(_shuffled(rng, [
                  Pred("operator", Ident(nm), Pred("cost", Num(ops[nm]["cost"])),
                       Pred("success_percent", Num(ops[nm]["p"])),
                       Pred("payoff_if_success", Num(ops[nm]["payoff"])),
                       Pred("payoff_if_failure", Num(ops[nm]["fallback"]))) for nm in names])),
              budget=Num(budget),
              query=Ident("highest_expected_value_operator"))
    return (obs, _shuffled(rng, names), best,
            {"answer": best, "greedy_answer": nbest, "budget": budget,
             "expected_values": {nm: ev[nm] / 100 for nm in names}})


class Metareasoning(Lesson):
    """Choosing how to reason, fallbacks included."""

    id = "metareasoning"
    level = 110
    tags = ("problem-formulation", "hierarchical-agency")
    teaches = "choosing how to reason, fallbacks included"
    capabilities = ('metareasoning', 'decision_theory', 'expected_value')
    axes = {'reasoning_depth': 5, 'computational_budget': 4, 'uncertainty': 4}

    generate = staticmethod(gen_metareasoning)
