"""``decomposition`` — a subgoal ordering that respects dependencies.

Problem formulation and hierarchical agency.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.epistemics import _labelled, _nonces, _shuffled, _topo_ok


def gen_decomposition(rng: random.Random, ctx):
    """Which ordering of the subgoals actually respects the dependencies.

    The dependency edges are given but the orderings are not sorted for you; one
    candidate is a genuine topological order and the other three each violate at
    least one edge. Validity is checked edge by edge, and the episode is
    rejected unless exactly one candidate passes.
    """
    tries = ctx.at(60, 600, default=60)           # a topological order is rarer when n grows
    for _ in range(400):
        n = rng.randint(*ctx.span((5, 6), (7, 8)))
        subs = _nonces(rng, n, 4)
        edges = [(subs[i], subs[j]) for i in range(n) for j in range(i + 1, n)
                 if rng.random() < 0.4]
        if len(edges) < 3:
            continue
        good = None
        for _ in range(tries):
            o = _shuffled(rng, subs)
            if _topo_ok(o, edges):
                good = o
                break
        if good is None:
            continue
        bad: list[list[str]] = []
        for _ in range(120):
            o = _shuffled(rng, subs)
            if not _topo_ok(o, edges) and o not in bad:
                bad.append(o)
            if len(bad) == 3:
                break
        if len(bad) == 3:
            break
    else:                                     # pragma: no cover - construction
        pass

    cands = [good] + bad
    labs, answer = _labelled(rng, cands, 0)
    entries = [Pred("plan", Ident(lab), Lst([Ident(s) for s in o]))
               for lab, o in zip(labs, cands)]
    obs = Rec(subgoals=Lst([Pred("subgoal", Ident(s)) for s in _shuffled(rng, subs)]),
              dependencies=Lst(_shuffled(rng, [Pred("before", Ident(a), Ident(b))
                                               for a, b in edges])),
              candidates=Lst(_shuffled(rng, entries)),
              query=Ident("ordering_that_respects_dependencies"))
    return (obs, _shuffled(rng, labs), answer,
            {"answer": answer, "n_edges": len(edges), "n_subgoals": n})


class Decomposition(Lesson):
    """A subgoal ordering that respects dependencies."""

    id = "decomposition"
    level = 105
    tags = ("problem-formulation", "hierarchical-agency")
    teaches = "a subgoal ordering that respects dependencies"
    capabilities = ('decomposition', 'planning', 'dependency_reasoning')
    axes = {'reasoning_depth': 3, 'planning_horizon': 3, 'world_complexity': 3}

    generate = staticmethod(gen_decomposition)
