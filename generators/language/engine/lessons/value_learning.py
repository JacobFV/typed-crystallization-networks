"""``value_learning`` — recover a preference order from observed choices.

Values and goal cognition.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.selfmodel import _labels, _rules, _shuffled


def gen_value_learning(rng: random.Random, ctx):
    """Recover the preference ordering behind a set of observed choices.

    The choices are generated from a hidden strict order and shown as unordered
    pairwise decisions, with a redundant non-adjacent choice thrown in. The
    ordering is fully determined but only through transitivity, so no single
    observation answers the question.
    """
    n = ctx.at(4, 8, default=4)
    ids = _labels(rng, "opt", n)
    order = list(rng.sample(range(n), n))         # order[0] is most preferred
    comps = [(order[i], order[i + 1]) for i in range(n - 1)]
    extra = [(order[i], order[j]) for i in range(n) for j in range(i + 2, n)]
    comps += rng.sample(extra, min(2, len(extra)))
    k = rng.randrange(n)
    facts = [Pred("chose", Ident(ids[a]), Ident(ids[b])) for a, b in _shuffled(rng, comps)]
    obs = Rec(choices=Lst(facts),
              rules=_rules("chose_a_b_means_a_is_strictly_preferred_to_b",
                           "preference_is_a_strict_total_order_and_is_transitive",
                           "rank_1_is_the_most_preferred_option"),
              query=Pred("rank", Num(k + 1)))
    return (obs, _shuffled(rng, ids), ids[order[k]],
            {"order": [ids[i] for i in order], "rank": k + 1})


class ValueLearning(Lesson):
    """Recover a preference order from observed choices."""

    id = "value_learning"
    level = 157
    tags = ("values", "goals")
    teaches = "recover a preference order from observed choices"
    capabilities = ('belief_modeling', 'value_learning')
    axes = {'reasoning_depth': 4, 'compositional_depth': 3, 'world_complexity': 2}

    generate = staticmethod(gen_value_learning)
