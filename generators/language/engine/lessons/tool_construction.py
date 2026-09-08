"""``tool_construction`` — build the tool that makes the workload cheapest.

Self-modeling and architecture adaptation.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.selfmodel import _labels, _rules, _shuffled


def gen_tool_construction(rng: random.Random, ctx):
    """Build which tool? The one that makes the stated future workload cheapest.

    Building has an up-front price, so a tool that covers the rarest query kind
    can lose to one that covers a common kind badly. The total is fully specified
    by the workload and the tool table, so "worth building" is an arithmetic
    fact.
    """
    n = ctx.at(4, 8, default=4)                  # query kinds, and tools to price
    kinds = _labels(rng, "kind", n)
    counts = [rng.randint(2, 9) for _ in range(n)]
    base = [rng.randint(4, 10) for _ in range(n)]
    ids = _labels(rng, "tool", n)
    for _ in range(80):
        tools = []
        for _i in range(n):
            cover = sorted(rng.sample(range(n), rng.randint(1, 2)))
            tools.append((cover, rng.randint(5, 45), rng.randint(1, 3)))
        totals = [b + sum(counts[k] * (c if k in cov else base[k]) for k in range(n))
                  for cov, b, c in tools]
        if sorted(totals)[0] != sorted(totals)[1]:
            break
    best = min(range(n), key=lambda i: totals[i])

    facts = [Pred("workload", Ident(kinds[k]), Num(counts[k]), Num(base[k])) for k in range(n)]
    tfacts = [Pred("tool", Ident(ids[i]), Num(tools[i][1]), Num(tools[i][2])) for i in range(n)]
    tfacts += [Pred("covers", Ident(ids[i]), Ident(kinds[k])) for i in range(n) for k in tools[i][0]]
    obs = Rec(workload=Lst(_shuffled(rng, facts)),
              tools=Lst(_shuffled(rng, tfacts)),
              rules=_rules("workload_lists_kind_count_and_cost_per_query_without_a_tool",
                           "tool_lists_build_cost_and_cost_per_covered_query",
                           "total_cost_is_build_cost_plus_sum_over_kinds_of_count_times_per_query_cost"),
              query=Ident("cheapest_tool_to_build"))
    return (obs, _shuffled(rng, ids), ids[best],
            {"totals": {ids[i]: totals[i] for i in range(n)}, "best_total": totals[best]})


class ToolConstruction(Lesson):
    """Build the tool that makes the workload cheapest."""

    id = "tool_construction"
    level = 141
    tags = ("self-modeling", "architecture")
    teaches = "build the tool that makes the workload cheapest"
    capabilities = ('architecture_adaptation', 'abstraction', 'planning')
    axes = {'reasoning_depth': 4, 'world_complexity': 3, 'compositional_depth': 3}

    generate = staticmethod(gen_tool_construction)
