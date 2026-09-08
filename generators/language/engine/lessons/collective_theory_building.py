"""``collective_theory_building`` — merge partially conflicting hypotheses into their consensus.

Protocols, institutions, and distributed intelligence.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.base import NAMES
from ..generators.reflective import _nonces, _shuffled


def gen_collective_theory_building(rng: random.Random, ctx):
    """Merge partially conflicting theories into the consensus they imply.

    Three researchers publish partial assignments over the same variables and
    disagree. The merge rule is stated: majority among those who took a stand,
    ties broken by seniority. The queried variable always carries a genuine
    conflict, so the consensus value has to be computed rather than copied.
    """
    fallback = None
    n_r = ctx.at(3, 6, default=3)
    for _ in range(300):
        variables = _nonces(rng, 3, 3)
        values = _nonces(rng, 4, 2)
        researchers = rng.sample(NAMES, n_r)
        rank = dict(zip(_shuffled(rng, researchers), list(range(1, n_r + 1))))
        claims: list[tuple[str, str, str]] = []
        for r in researchers:
            for v in rng.sample(variables, rng.randint(2, 3)):
                claims.append((r, v, rng.choice(values)))
        contested = [v for v in variables
                     if len({c[2] for c in claims if c[1] == v}) > 1
                     and sum(1 for c in claims if c[1] == v) >= 2]
        if not contested:
            continue
        var = rng.choice(contested)
        rel = [c for c in claims if c[1] == var]
        tally: dict[str, int] = {}
        for _, _, val in rel:
            tally[val] = tally.get(val, 0) + 1
        best = max(tally.values())
        tied = [v for v, c in tally.items() if c == best]
        if len(tied) == 1:
            answer = tied[0]
        else:
            answer = min([c for c in rel if c[2] in tied], key=lambda c: rank[c[0]])[2]
        fallback = (variables, values, researchers, rank, claims, var, answer)
        break
    variables, values, researchers, rank, claims, var, answer = fallback
    obs = Rec(researchers=Lst([Pred("researcher", Ident(r), Ident("seniority"), Num(rank[r]))
                              for r in researchers]),
              hypotheses=Lst(_shuffled(rng, [Pred("claims", Ident(r), Ident(v), Ident(val))
                                             for r, v, val in claims])),
              merge_rule=Pred("consensus", Ident("majority_then_seniority")),
              variables=Lst([Ident(v) for v in variables]),
              query=Pred("consensus_value_of", Ident(var)))
    return obs, _shuffled(rng, values), answer, {"variable": var, "consensus": answer,
                                                 "n_claims": len(claims)}


class CollectiveTheoryBuilding(Lesson):
    """Merge partially conflicting hypotheses into their consensus."""

    id = "collective_theory_building"
    level = 129
    tags = ("protocols", "institutions", "distributed-intelligence")
    teaches = "merge partially conflicting hypotheses into their consensus"
    capabilities = ('scientific_induction', 'ontology_learning', 'multi_agent_coordination')
    axes = {'reasoning_depth': 4, 'world_complexity': 4, 'lexical_novelty': 4}

    generate = staticmethod(gen_collective_theory_building)
