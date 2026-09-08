"""``deception_detection`` — statement content vs truth vs incentive.

Analogy, causality, planning, and programs.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec, Term
from ..lesson import Lesson
from ..generators.base import NAMES
from ..generators.social import ITEMS, PLACES, _shuffled


def gen_deception_detection(rng: random.Random, ctx):
    """One speaker's claim contradicts the world; incentive alone cannot say who.

    Every speaker reports the location of a different item, and exactly one report
    is false *by construction* — the world was generated first and the liar's
    claim was then moved off it. Two speakers carry a motive to conceal, so the
    incentive facts halve the hypothesis space and no more: the remaining step is
    to check each claim against the world state.
    """
    n = ctx.at(4, 6, default=4)
    speakers = rng.sample(NAMES, n)
    items = rng.sample(ITEMS, n)
    world = {it: rng.choice(PLACES) for it in items}
    liar_ix = rng.randrange(n)
    liar = speakers[liar_ix]
    claims: list[Term] = []
    claimed: dict[str, str] = {}
    for i, sp in enumerate(speakers):
        it = items[i]
        place = world[it]
        if i == liar_ix:
            place = rng.choice([p for p in PLACES if p != world[it]])
        claimed[sp] = place
        claims.append(Pred("claims", Ident(sp), Pred("at", Ident(it), Ident(place))))
    honest_with_motive = rng.choice([s for s in speakers if s != liar])
    motives = [Pred("motive", Ident(s),
                    Ident("conceal" if s in (liar, honest_with_motive) else "none"))
               for s in speakers]
    obs = Rec(world=Lst(_shuffled(rng, [Pred("at", Ident(it), Ident(world[it])) for it in items])),
              motives=Lst(_shuffled(rng, motives)),
              testimony=Lst(_shuffled(rng, claims)),
              query=Ident("who_lied"))
    return (obs, _shuffled(rng, speakers), liar,
            {"liar": liar, "decoy_motive": honest_with_motive,
             "world": dict(world), "claimed": claimed})


class DeceptionDetection(Lesson):
    """Statement content vs truth vs incentive."""

    id = "deception_detection"
    level = 50
    tags = ("analogy", "causality", "planning", "programs")
    teaches = "statement content vs truth vs incentive"
    capabilities = ('belief_modeling', 'multi_agent_coordination')
    axes = {'reasoning_depth': 3, 'world_complexity': 3, 'ambiguity': 2}

    generate = staticmethod(gen_deception_detection)
