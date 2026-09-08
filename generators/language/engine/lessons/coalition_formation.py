"""``coalition_formation`` — which grouping no subset of agents wants to break.

Protocols, institutions, and distributed intelligence.
"""

from __future__ import annotations

import random
from typing import Sequence

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.reflective import _labels, _partitions, _shuffled


def gen_coalition_formation(rng: random.Random, ctx):
    """Which grouping is stable when any subset may walk out together?

    Every coalition has a value and members split it equally, so each candidate
    partition gives each agent an exact share. A partition is stable when no
    subset of agents — of the fifteen possible — can form and leave every one of
    its members strictly better off. All fifteen partitions of four agents are
    checked; the option set is built with one stable partition and three
    provably blockable ones.
    """
    agents = ["a", "b", "c", "d", "e", "f"][:ctx.at(4, 6, default=4)]
    subsets = [tuple(s for s in agents if (i >> agents.index(s)) & 1)
               for i in range(1, 2 ** len(agents))]
    fallback = None
    for _ in range(300):
        share = {s: rng.randint(1, 9) for s in subsets}         # per-member payoff of s
        parts = _partitions(agents)

        def stable(part: Sequence[Sequence[str]]) -> bool:
            cur = {x: share[tuple(sorted(b))] for b in part for x in b}
            return not any(all(share[s] > cur[x] for x in s) for s in subsets)

        good = [p for p in parts if stable(p)]
        bad = [p for p in parts if not stable(p)]
        cand = None
        if good and len(bad) >= 3:
            options = [rng.choice(good)] + rng.sample(bad, 3)
            ids = _labels(rng, "g", 4)
            order = _shuffled(rng, range(4))
            assign = {ids[k]: options[order[k]] for k in range(4)}
            answer = [i for i in ids if assign[i] in good][0]
            cand = (share, assign, ids, answer, len(good))
        if cand is None:
            continue
        fallback = cand
        break
    share, assign, ids, answer, n_stable = fallback
    obs = Rec(values=Lst([Pred("coalition", Lst([Ident(x) for x in s]),
                               Num(share[s] * len(s))) for s in subsets]),
              split_rule=Pred("members_split", Ident("equally")),
              stability=Pred("stable_if", Pred("no_subset_makes_all_its_members_better")),
              options=Lst(_shuffled(rng, [Pred("grouping", Ident(i),
                                               Lst([Lst([Ident(x) for x in b]) for b in assign[i]]))
                                          for i in ids])),
              query=Ident("which_grouping_is_stable"))
    return (obs, _shuffled(rng, ids), answer,
            {"stable": [list(map(list, assign[answer]))], "n_stable_partitions": n_stable})


class CoalitionFormation(Lesson):
    """Which grouping no subset of agents wants to break."""

    id = "coalition_formation"
    level = 127
    tags = ("protocols", "institutions", "distributed-intelligence")
    teaches = "which grouping no subset of agents wants to break"
    capabilities = ('multi_agent_coordination', 'planning', 'abstraction')
    axes = {'reasoning_depth': 5, 'world_complexity': 5, 'compositional_depth': 4}

    generate = staticmethod(gen_coalition_formation)
