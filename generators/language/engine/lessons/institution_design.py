"""``institution_design`` — which rule makes the target collective outcome an equilibrium.

Protocols, institutions, and distributed intelligence.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.base import NAMES
from ..generators.reflective import _labels, _pure_equilibria, _shuffled


def gen_institution_design(rng: random.Random, ctx):
    """Which rule makes the collective outcome happen?

    Three agents with different private costs choose contribute/defect in a
    public-goods game; each candidate rule bolts a fine, a subsidy or a
    threshold bonus onto the payoffs. Every candidate is *simulated*: the whole
    3-player game is enumerated and its pure Nash equilibria found, and the
    world is kept only when each rule has a unique equilibrium and exactly one
    of them produces the stated number of contributors.
    """
    fallback = None
    n_a = ctx.at(3, 5, default=3)
    for _ in range(500):
        agents = rng.sample(NAMES, n_a)
        cost = {a: rng.randint(1, 6) for a in agents}
        benefit = rng.randint(1, 3)
        ids = _labels(rng, "r", 4)
        specs = [("none", 0, 0), ("fine", rng.randint(1, 6), 0),
                 ("subsidy", rng.randint(1, 6), 0),
                 ("threshold", rng.randint(2, 3), rng.randint(2, 8))]
        spec = dict(zip(ids, _shuffled(rng, specs)))

        def make(kind: str, p1: int, p2: int):
            def payoff(prof: tuple[int, ...], i: int) -> int:
                k = sum(prof)
                v = benefit * k - (cost[agents[i]] if prof[i] else 0)
                if kind == "fine" and not prof[i]:
                    v -= p1
                if kind == "subsidy" and prof[i]:
                    v += p1
                if kind == "threshold" and k >= p1:
                    v += p2
                return v
            return payoff

        eqs = {i: _pure_equilibria(n_a, make(*spec[i])) for i in ids}
        if any(len(e) != 1 for e in eqs.values()):
            continue
        counts = {i: sum(eqs[i][0]) for i in ids}
        target = rng.choice(sorted(set(counts.values())))
        winners = [i for i in ids if counts[i] == target]
        cand = (agents, cost, benefit, ids, spec, counts, target, winners[0])
        if fallback is None:
            fallback = cand
        if len(winners) == 1:
            fallback = cand
            break
    agents, cost, benefit, ids, spec, counts, target, answer = fallback
    obs = Rec(agents=Lst([Pred("agent", Ident(a), Pred("contribution_cost"), Num(cost[a]))
                          for a in agents]),
              payoff=Pred("each_agent_gains", Pred("benefit_per_contributor"), Num(benefit)),
              rules=Lst(_shuffled(rng, [Pred("rule", Ident(i), Ident(spec[i][0]),
                                             Num(spec[i][1]), Num(spec[i][2])) for i in ids])),
              objective=Pred("contributors_in_equilibrium", Num(target)),
              query=Ident("which_rule"))
    return obs, _shuffled(rng, ids), answer, {"equilibrium_counts": counts, "target": target,
                                              "costs": cost, "benefit": benefit}


class InstitutionDesign(Lesson):
    """Which rule makes the target collective outcome an equilibrium."""

    id = "institution_design"
    level = 123
    tags = ("protocols", "institutions", "distributed-intelligence")
    teaches = "which rule makes the target collective outcome an equilibrium"
    capabilities = ('multi_agent_coordination', 'planning', 'metareasoning')
    axes = {'reasoning_depth': 5, 'world_complexity': 4, 'compositional_depth': 3}

    generate = staticmethod(gen_institution_design)
