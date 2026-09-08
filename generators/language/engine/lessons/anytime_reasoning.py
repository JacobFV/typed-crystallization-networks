"""``anytime_reasoning`` — quality-vs-compute profiles read at a deadline.

Problem formulation and hierarchical agency.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.epistemics import _nonces, _shuffled


def gen_anytime_reasoning(rng: random.Random, ctx):
    """Quality against compute, read at a deadline.

    Every algorithm publishes a profile of (compute, quality) checkpoints; at a
    deadline its quality is whatever its last reached checkpoint delivered, and
    nothing at all if it has not reached one. The profiles are built to cross,
    and the episode is rejected unless the winner at the deadline differs from
    the winner given unlimited time — the point being that the best algorithm is
    a function of the budget.
    """
    for _ in range(400):
        names = _nonces(rng, ctx.at(4, 8, default=4), 4)
        deadline = rng.randint(4, 14)
        profiles: dict[str, list[tuple[int, int]]] = {}
        for nm in names:
            ts = sorted(rng.sample(range(1, 20), rng.randint(3, 4)))
            qs = sorted(rng.sample(range(5, 100), len(ts)))
            profiles[nm] = list(zip(ts, qs))

        def at(nm: str, t: int) -> int:
            q = 0
            for tt, qq in profiles[nm]:
                if tt <= t:
                    q = qq
            return q

        now = {nm: at(nm, deadline) for nm in names}
        end = {nm: at(nm, 100) for nm in names}
        best = max(names, key=lambda nm: now[nm])
        ebest = max(names, key=lambda nm: end[nm])
        if list(now.values()).count(now[best]) != 1 or now[best] == 0:
            continue
        if list(end.values()).count(end[ebest]) != 1:
            continue
        if best != ebest:
            break
    else:                                     # pragma: no cover - construction
        pass

    obs = Rec(profiles=Lst(_shuffled(rng, [
                  Pred("profile", Ident(nm),
                       Lst([Pred("at", Num(t), Num(q)) for t, q in profiles[nm]]))
                  for nm in names])),
              query=Pred("best_answer_by_deadline", Num(deadline)))
    return (obs, _shuffled(rng, names), best,
            {"deadline": deadline, "answer": best, "best_unbounded": ebest,
             "quality_at_deadline": now})


class AnytimeReasoning(Lesson):
    """Quality-vs-compute profiles read at a deadline."""

    id = "anytime_reasoning"
    level = 109
    tags = ("problem-formulation", "hierarchical-agency")
    teaches = "quality-vs-compute profiles read at a deadline"
    capabilities = ('metareasoning', 'decision_theory', 'computational_cost')
    axes = {'reasoning_depth': 3, 'computational_budget': 4, 'world_complexity': 3}

    generate = staticmethod(gen_anytime_reasoning)
