"""``problem_reformulation`` — the encoding whose search tree is smallest.

Problem formulation and hierarchical agency.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.epistemics import _backtrack, _labelled, _nonces, _shuffled


def gen_problem_reformulation(rng: random.Random, ctx):
    """Same problem, four encodings, very different search costs.

    The constraint network is fixed; what varies is the order the variables are
    instantiated in. Each candidate order is run through the stated procedure
    (chronological backtracking, ascending values, first solution) and the node
    counts are compared exactly, so the answer is a property of the search tree
    rather than of how the ordering looks.
    """
    for _ in range(400):
        nv = rng.randint(*ctx.span((4, 5), (6, 7)))
        vars_ = _nonces(rng, nv, 4)
        domains = {v: sorted(rng.sample(range(1, 7), rng.randint(2, 4))) for v in vars_}
        cons: list[tuple[str, str, str]] = []
        for i in range(nv):
            for j in range(i + 1, nv):
                if rng.random() < 0.55:
                    cons.append((vars_[i], rng.choice(["!=", "<", ">"]), vars_[j]))
        if not cons:
            continue
        orders: list[list[str]] = []
        for _ in range(60):
            o = _shuffled(rng, vars_)
            if o not in orders:
                orders.append(o)
            if len(orders) == 4:
                break
        costs, oks = [], []
        for o in orders:
            n, ok = _backtrack(o, domains, cons)
            costs.append(n)
            oks.append(ok)
        if not all(oks):
            continue
        best = min(costs)
        if costs.count(best) == 1 and max(costs) - best >= 3:
            break
    else:                                     # pragma: no cover - construction
        pass

    labs, answer = _labelled(rng, orders, costs.index(best))
    entries = [Pred("encoding", Ident(lab), Lst([Ident(v) for v in o]))
               for lab, o in zip(labs, orders)]
    obs = Rec(variables=Lst([Pred("domain", Ident(v), Lst([Num(d) for d in domains[v]]))
                             for v in vars_]),
              constraints=Lst(_shuffled(rng, [Pred("constrain", Ident(a), Ident(r), Ident(b))
                                              for a, r, b in cons])),
              procedure=Pred("search", Ident("backtracking"), Pred("values_ascending"),
                             Pred("stop_at_first_solution"), Pred("cost_is_assignments_tried")),
              candidates=Lst(_shuffled(rng, entries)),
              query=Ident("cheapest_encoding"))
    return (obs, _shuffled(rng, labs), answer,
            {"costs": dict(zip(labs, costs)), "answer": answer, "n_constraints": len(cons)})


class ProblemReformulation(Lesson):
    """The encoding whose search tree is smallest."""

    id = "problem_reformulation"
    level = 104
    tags = ("problem-formulation", "hierarchical-agency")
    teaches = "the encoding whose search tree is smallest"
    capabilities = ('problem_formulation', 'search', 'computational_cost')
    axes = {'reasoning_depth': 5, 'computational_budget': 3, 'world_complexity': 3}

    generate = staticmethod(gen_problem_reformulation)
