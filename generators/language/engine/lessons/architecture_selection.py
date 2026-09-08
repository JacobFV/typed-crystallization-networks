"""``architecture_selection`` — cheapest architecture that covers the task.

Self-modeling and architecture adaptation.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.selfmodel import FEATURES, _labels, _rules, _shuffled


def gen_architecture_selection(rng: random.Random, ctx):
    """Cheapest architecture that actually covers the task's requirements.

    The globally cheapest architecture is always missing a requirement, so
    reading off the smallest number is wrong every time: feasibility has to be
    checked first and cost only afterwards.
    """
    n = ctx.at(4, 9, default=4)
    for _ in range(60):
        req = rng.sample(FEATURES, 2)
        costs = rng.sample(range(3, 24), n)
        order = sorted(range(n), key=lambda i: costs[i])
        feats: dict[int, list[str]] = {}
        for i in range(n):
            ok = (i != order[0]) and (i == order[1] or rng.random() < 0.5)
            others = [f for f in FEATURES if f not in req]
            if ok:
                feats[i] = _shuffled(rng, req + rng.sample(others, rng.randint(0, 2)))
            else:
                keep = [f for f in req if rng.random() < 0.5]
                if len(keep) == len(req):
                    keep = keep[:-1]
                feats[i] = _shuffled(rng, keep + rng.sample(others, rng.randint(1, 3)))
        ok_idx = [i for i in range(n) if set(req) <= set(feats[i])]
        if ok_idx and min(ok_idx, key=lambda i: costs[i]) == order[1] and order[0] not in ok_idx:
            break

    ids = _labels(rng, "arch", n)
    best = order[1]
    facts = [Pred("architecture", Ident(ids[i]), Num(costs[i])) for i in range(n)]
    facts += [Pred("provides", Ident(ids[i]), Ident(f)) for i in range(n) for f in feats[i]]
    obs = Rec(architectures=Lst(_shuffled(rng, facts)),
              task=Lst([Pred("needs", Ident(f)) for f in _shuffled(rng, req)]),
              rules=_rules("an_architecture_can_run_the_task_iff_it_provides_every_needed_feature",
                           "choose_the_runnable_architecture_of_least_cost"),
              query=Ident("which_architecture"))
    return (obs, _shuffled(rng, ids), ids[best],
            {"needs": req, "costs": {ids[i]: costs[i] for i in range(n)},
             "feasible": [ids[i] for i in ok_idx]})


class ArchitectureSelection(Lesson):
    """Cheapest architecture that covers the task."""

    id = "architecture_selection"
    level = 139
    tags = ("self-modeling", "architecture")
    teaches = "cheapest architecture that covers the task"
    capabilities = ('architecture_adaptation', 'metareasoning')
    axes = {'reasoning_depth': 3, 'world_complexity': 3, 'compositional_depth': 2}

    generate = staticmethod(gen_architecture_selection)
