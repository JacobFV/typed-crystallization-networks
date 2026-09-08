"""``negotiation_game`` — speech acts under private utilities.

Analogy, causality, planning, and programs.
"""

from __future__ import annotations

import random
from typing import Mapping

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.social import ITEMS, TAGS, _shuffled


def gen_negotiation_game(rng: random.Random, ctx):
    """Alternating offers under stated reservation values, simulated to a close.

    Each party values the items differently and accepts the first offer worth at
    least its reservation value. The transcript is a sequence of proposals; the
    question is which one closed the deal — which requires scoring every offer
    from the *responder's* side, not the proposer's, and stopping at the first
    acceptance rather than at the best offer on the table. Deadlock ("none") is
    one of the four outcomes and is drawn as often as any other.
    """
    items = rng.sample(ITEMS, ctx.at(3, 6, default=3))
    parties = ["a", "b"]
    for _ in range(400):
        values = {p: {it: rng.randint(1, 6) for it in items} for p in parties}
        offers = [{it: rng.choice(parties) for it in items} for _ in range(3)]
        opener = rng.choice(parties)
        proposer = [opener if k % 2 == 0 else parties[1 - parties.index(opener)] for k in range(3)]
        responder = [parties[1 - parties.index(p)] for p in proposer]

        def util(p: str, offer: Mapping[str, str]) -> int:
            return sum(values[p][it] for it in items if offer[it] == p)

        target = rng.randrange(4)                     # 3 == deadlock
        rejected = {p: [util(p, offers[k]) for k in range(min(target, 3)) if responder[k] == p]
                    for p in parties}
        thresholds: dict[str, int] = {}
        if target < 3:
            acc = responder[target]
            acc_val = util(acc, offers[target])
            if rejected[acc] and acc_val <= max(rejected[acc]):
                continue                              # cannot both reject and accept
            thresholds[acc] = acc_val
            other = parties[1 - parties.index(acc)]
            thresholds[other] = (max(rejected[other]) + 1) if rejected[other] else rng.randint(1, 8)
        else:
            if not all(rejected[p] for p in parties):
                continue
            for p in parties:
                thresholds[p] = max(rejected[p]) + 1

        # simulate the protocol rather than trusting the construction
        closed: int | None = None
        for k in range(3):
            if util(responder[k], offers[k]) >= thresholds[responder[k]]:
                closed = k
                break
        if (closed if closed is not None else 3) != target:
            continue

        tags = rng.sample(TAGS, 3)
        value_facts = [Pred("value", Ident(p), Ident(it), Num(values[p][it]))
                       for p in parties for it in items]
        rules = [Pred("accepts_if_at_least", Ident(p), Num(thresholds[p])) for p in parties]
        transcript = [Pred("offer", Ident(tags[k]), Ident(proposer[k]),
                           Lst([Pred("gets", Ident(offers[k][it]), Ident(it)) for it in items]))
                      for k in range(3)]
        obs = Rec(values=Lst(_shuffled(rng, value_facts)), rules=Lst(_shuffled(rng, rules)),
                  transcript=Lst(transcript), query=Ident("accepted_offer"))
        answer = tags[target] if target < 3 else "none"
        return (obs, _shuffled(rng, tags + ["none"]), answer,
                {"thresholds": dict(thresholds), "closed_at": target,
                 "utilities": {p: [util(p, o) for o in offers] for p in parties}})
    raise RuntimeError("negotiation_game: no admissible world")


class NegotiationGame(Lesson):
    """Speech acts under private utilities."""

    id = "negotiation_game"
    level = 49
    tags = ("analogy", "causality", "planning", "programs")
    teaches = "speech acts under private utilities"
    capabilities = ('multi_agent_coordination', 'planning')
    axes = {'discourse_horizon': 3, 'reasoning_depth': 4, 'world_complexity': 3}

    generate = staticmethod(gen_negotiation_game)
