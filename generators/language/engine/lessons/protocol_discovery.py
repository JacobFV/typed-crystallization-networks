"""``protocol_discovery`` — which interaction discipline is consistent with every trace.

Protocols, institutions, and distributed intelligence.
"""

from __future__ import annotations

import random
from typing import Sequence

from .._structure import Ident, Lst, Num, Pred, Rec, Term
from ..lesson import Lesson
from ..generators.base import NAMES
from ..generators.reflective import _labels, _shuffled


def gen_protocol_discovery(rng: random.Random, ctx):
    """Which service discipline governs these interaction traces?

    A server answers requests in *some* order. The candidate protocols are
    given parametrically — sort by arrival / priority / name, ascending or
    descending — so their names are opaque labels and only the traces identify
    the rule. The generator keeps a world only when exactly one candidate
    reproduces every trace, so "consistent with the evidence" is a fact.
    """
    keys = [("arrival", "asc"), ("arrival", "desc"), ("priority", "asc"),
            ("priority", "desc"), ("name", "asc")]

    def order(reqs: Sequence[tuple[str, int, int]], key: str, direction: str) -> list[str]:
        idx = {"arrival": 2, "priority": 1, "name": 0}[key]
        return [r[0] for r in sorted(reqs, key=lambda r: r[idx], reverse=(direction == "desc"))]

    fallback = None
    for _ in range(300):
        ids = _labels(rng, "p", len(keys))
        spec = dict(zip(ids, keys))
        truth = rng.choice(ids)
        traces = []
        for _ in range(3):
            n = rng.randint(*ctx.span((3, 4), (5, 6)))    # requests per trace
            clients = rng.sample(NAMES, n)
            prios = rng.sample(range(1, 10), n)
            reqs = [(clients[i], prios[i], i) for i in range(n)]
            traces.append((reqs, order(reqs, *spec[truth])))
        ok = [i for i in ids if all(order(r, *spec[i]) == resp for r, resp in traces)]
        cand = (ids, spec, truth, traces)
        if fallback is None:
            fallback = cand
        if ok == [truth]:
            fallback = cand
            break
    ids, spec, truth, traces = fallback
    facts: list[Term] = []
    for ti, (reqs, resp) in enumerate(traces):
        for c, pr, arr in reqs:
            facts.append(Pred("request", Num(ti), Num(arr), Ident(c), Num(pr)))
        for pos, c in enumerate(resp):
            facts.append(Pred("response", Num(ti), Num(pos), Ident(c)))
    obs = Rec(traces=Lst(facts),
              protocols=Lst(_shuffled(rng, [Pred("protocol", Ident(i), Ident(spec[i][0]),
                                                 Ident(spec[i][1])) for i in ids])),
              semantics=Pred("responses_sorted_by", Ident("key"), Ident("direction")),
              query=Ident("which_protocol"))
    return obs, _shuffled(rng, ids), truth, {"protocol": list(spec[truth]), "n_traces": len(traces)}


class ProtocolDiscovery(Lesson):
    """Which interaction discipline is consistent with every trace."""

    id = "protocol_discovery"
    level = 121
    tags = ("protocols", "institutions", "distributed-intelligence")
    teaches = "which interaction discipline is consistent with every trace"
    capabilities = ('multi_agent_coordination', 'scientific_induction', 'temporal_reasoning')
    axes = {'discourse_horizon': 4, 'reasoning_depth': 4, 'world_complexity': 4}

    generate = staticmethod(gen_protocol_discovery)
