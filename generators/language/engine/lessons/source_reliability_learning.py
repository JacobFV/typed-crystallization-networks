"""``source_reliability_learning`` — conditional, per-domain reliability.

Epistemics, argument, and teaching.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.epistemics import DOMAINS, SOURCES, _shuffled


def gen_source_reliability_learning(rng: random.Random, ctx):
    """Reliability is conditional, not a scalar.

    Each source has a *per-domain* accuracy, shown only as an itemized track
    record of past adjudicated claims. The query names a domain and asks who to
    trust there. Trial counts differ across sources, so the raw number of
    correct calls is not the answer; and in half the episodes the best source
    overall is deliberately not the best source in the queried domain, which is
    exactly the distinction a single global score cannot make.
    """
    srcs = rng.sample(SOURCES, ctx.at(4, 8, default=4))
    domains = rng.sample(DOMAINS, 3)
    qd = rng.choice(domains)
    want_split = rng.random() < 0.5           # force local != global half the time

    rec: dict[tuple[str, str], tuple[int, int]] = {}
    for _ in range(200):
        for s in srcs:
            for d in domains:
                n = rng.randint(4, 9)
                rec[(s, d)] = (rng.randint(0, n), n)
        # unique argmax by accuracy in the queried domain (exact rational compare)
        best = max(srcs, key=lambda s: (rec[(s, qd)][0] / rec[(s, qd)][1], s))
        bk, bn = rec[(best, qd)]
        if sum(1 for s in srcs if rec[(s, qd)][0] * bn == bk * rec[(s, qd)][1]) != 1:
            continue
        tot = {s: (sum(rec[(s, d)][0] for d in domains), sum(rec[(s, d)][1] for d in domains))
               for s in srcs}
        gbest = max(srcs, key=lambda s: (tot[s][0] / tot[s][1], s))
        gk, gn = tot[gbest]
        if sum(1 for s in srcs if tot[s][0] * gn == gk * tot[s][1]) != 1:
            continue
        if (gbest != best) == want_split:
            break

    trials = []
    i = 0
    for s in srcs:
        for d in domains:
            k, n = rec[(s, d)]
            for j in range(n):
                trials.append(Pred("adjudicated", Num(i), Ident(s), Ident(d),
                                   Ident("correct" if j < k else "wrong")))
                i += 1
    obs = Rec(track_record=Lst(_shuffled(rng, trials)),
              domains=Lst([Ident(d) for d in sorted(domains)]),
              query=Pred("most_reliable_in", Ident(qd)))
    return (obs, _shuffled(rng, srcs), best,
            {"domain": qd, "best_in_domain": best, "best_overall": gbest,
             "accuracies": {f"{s}/{qd}": f"{rec[(s, qd)][0]}/{rec[(s, qd)][1]}" for s in srcs}})


class SourceReliabilityLearning(Lesson):
    """Conditional, per-domain reliability."""

    id = "source_reliability_learning"
    level = 95
    tags = ("epistemics", "argument", "teaching")
    teaches = "conditional, per-domain reliability"
    capabilities = ('source_modelling', 'induction', 'calibration')
    axes = {'reasoning_depth': 3, 'world_complexity': 4, 'uncertainty': 3}

    generate = staticmethod(gen_source_reliability_learning)
