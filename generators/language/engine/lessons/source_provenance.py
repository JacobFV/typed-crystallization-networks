"""``source_provenance`` — what is established vs what was merely reported.

Epistemics, argument, and teaching.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.epistemics import PROVENANCE_LABELS, SOURCES, _nonces, _provenance_status, _shuffled


def gen_source_provenance(rng: random.Random, ctx):
    """``x said p`` is a separate symbol from ``p``.

    Two of the four sources are declared unreliable in the episode, so a report
    is evidence only when its origin is. The five statuses partition the cases:
    a proposition can be established, refuted, contested between reliable
    sources, merely reported by unreliable ones, or never mentioned at all —
    and only the last two are distinguishable by looking at who spoke.
    """
    label = rng.choice(PROVENANCE_LABELS)
    srcs = rng.sample(SOURCES, 4)
    reliable, unreliable = set(srcs[:2]), set(srcs[2:])
    props = _nonces(rng, ctx.at(4, 12, default=4), 4)
    q = props[0]

    reports: list[tuple[str, str, str]] = []
    for p in props[1:]:                       # background chatter about other claims
        for s in srcs:
            if rng.random() < 0.45:
                reports.append((s, p, rng.choice(["yes", "no"])))

    rel_list, unrel_list = sorted(reliable), sorted(unreliable)
    if label == "established":
        for s in rng.sample(rel_list, rng.randint(1, 2)):
            reports.append((s, q, "yes"))
        for s in unrel_list:
            if rng.random() < 0.5:
                reports.append((s, q, rng.choice(["yes", "no"])))
    elif label == "refuted":
        for s in rng.sample(rel_list, rng.randint(1, 2)):
            reports.append((s, q, "no"))
        for s in unrel_list:
            if rng.random() < 0.5:
                reports.append((s, q, rng.choice(["yes", "no"])))
    elif label == "contested":
        reports.append((rel_list[0], q, "yes"))
        reports.append((rel_list[1], q, "no"))
        for s in unrel_list:
            if rng.random() < 0.5:
                reports.append((s, q, rng.choice(["yes", "no"])))
    elif label == "reported_only":
        for s in rng.sample(unrel_list, rng.randint(1, 2)):
            reports.append((s, q, rng.choice(["yes", "no"])))
    # "unmentioned": nobody speaks about q at all

    truth = _provenance_status(reliable, unreliable, reports, q)   # recomputed, never assumed
    facts = ([Pred("reliable", Ident(s)) for s in rel_list]
             + [Pred("unreliable", Ident(s)) for s in unrel_list])
    obs = Rec(sources=Lst(_shuffled(rng, facts)),
              reports=Lst(_shuffled(rng, [Pred("said", Ident(s), Ident(p), Ident(pol))
                                          for s, p, pol in reports])),
              query=Pred("status_of", Ident(q)))
    return (obs, _shuffled(rng, PROVENANCE_LABELS), truth,
            {"label": truth, "proposition": q, "reliable": rel_list,
             "unreliable": unrel_list, "n_reports": len(reports)})


class SourceProvenance(Lesson):
    """What is established vs what was merely reported."""

    id = "source_provenance"
    level = 94
    tags = ("epistemics", "argument", "teaching")
    teaches = "what is established vs what was merely reported"
    capabilities = ('provenance', 'epistemic_status', 'source_modelling')
    axes = {'reasoning_depth': 3, 'world_complexity': 3, 'ambiguity': 2}
    answers = ['established', 'refuted', 'contested', 'reported_only', 'unmentioned']

    generate = staticmethod(gen_source_provenance)
