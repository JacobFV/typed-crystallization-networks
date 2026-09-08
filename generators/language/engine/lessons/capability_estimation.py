"""``capability_estimation`` — which task exceeds the described agent.

Self-modeling and architecture adaptation.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.selfmodel import SKILLS, _labels, _rules, _shuffled


def gen_capability_estimation(rng: random.Random, ctx):
    """Of four candidate tasks, exactly one is beyond the described agent.

    Estimating capability *before* acting is the point, so the question is which
    task to expect a failure on rather than what happened. One module is missing
    and the budget is finite; the single infeasible task trips one of those two,
    chosen at random, and the other three are affordable and covered.
    """
    skills = rng.sample(SKILLS, 4)
    missing = rng.choice(skills)
    present = [s for s in skills if s != missing]
    budget = rng.randint(8, 14)
    log = [Pred("attempt", Ident(f"log{i}"), Ident(s), Num(1),
                Ident("failed" if s == missing else "solved")) for i, s in enumerate(skills)]

    n_tasks = ctx.at(4, 9, default=4)            # candidate tasks to screen
    ids = _labels(rng, "task", n_tasks)
    fail_i = rng.randrange(n_tasks)
    by_budget = rng.random() < 0.5
    tasks: list[tuple[str, list[str], int]] = []
    for i in range(n_tasks):
        if i == fail_i and not by_budget:
            req = _shuffled(rng, [missing, rng.choice(present)])
            cost = rng.randint(2, budget)
        elif i == fail_i:
            req = rng.sample(present, 2)
            cost = rng.randint(budget + 1, budget + 6)
        else:
            req = rng.sample(present, 2)
            cost = rng.randint(2, budget)
        tasks.append((ids[i], req, cost))

    def feasible(t: tuple[str, list[str], int]) -> bool:
        return all(s != missing for s in t[1]) and t[2] <= budget

    assert sum(1 for t in tasks if not feasible(t)) == 1
    facts = [Pred("task", Ident(tid), Ident(r[0]), Ident(r[1]), Num(c)) for tid, r, c in tasks]
    obs = Rec(agent=Lst([Pred("budget", Num(budget))]),
              history=Lst(_shuffled(rng, log)),
              tasks=Lst(_shuffled(rng, facts)),
              rules=_rules("each_logged_attempt_used_one_module_and_cost_1",
                           "a_task_is_solved_iff_both_required_modules_are_present_and_cost_le_budget"),
              query=Ident("which_task_fails"))
    return (obs, _shuffled(rng, ids), ids[fail_i],
            {"missing": missing, "budget": budget, "mode": "budget" if by_budget else "module"})


class CapabilityEstimation(Lesson):
    """Which task exceeds the described agent."""

    id = "capability_estimation"
    level = 136
    tags = ("self-modeling", "architecture")
    teaches = "which task exceeds the described agent"
    capabilities = ('self_modeling', 'metareasoning')
    axes = {'reasoning_depth': 3, 'world_complexity': 3, 'compositional_depth': 2}

    generate = staticmethod(gen_capability_estimation)
