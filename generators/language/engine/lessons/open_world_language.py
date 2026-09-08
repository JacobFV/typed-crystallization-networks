"""``open_world_language`` — false vs unknown under a closed roster.

Analogy, causality, planning, and programs.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.base import NAMES
from ..generators.social import PROPS, _shuffled


def gen_open_world_language(rng: random.Random, ctx):
    """Closed world inside the roster, open world outside it.

    The roster names the entities whose records are complete, so absence of a fact
    about a rostered entity means *false* while a question about an unrostered
    entity means *unknown*. The three outcomes are drawn uniformly and the world
    is then built to produce the drawn one, so the epistemic distinction has to be
    made every episode and cannot be papered over with a default.
    """
    entities = rng.sample(NAMES, 5)
    roster, outside = entities[:3], entities[3:]
    props = rng.sample(PROPS, ctx.at(3, 6, default=3))
    # every rostered entity has at least one property and lacks at least one, so
    # all three answers are always available and the case can be drawn uniformly
    has = {e: rng.sample(props, rng.randint(1, 2)) for e in roster}
    case = rng.choice(["yes", "no", "unknown"])
    if case == "yes":
        pool = [(e, p) for e in roster for p in has[e]]
    elif case == "no":
        pool = [(e, p) for e in roster for p in props if p not in has[e]]
    else:
        pool = [(e, p) for e in outside for p in props]
    ent, prop = pool[rng.randrange(len(pool))]
    facts = [Pred("has", Ident(e), Ident(p)) for e in roster for p in has[e]]
    obs = Rec(roster=Lst([Pred("recorded", Ident(e)) for e in _shuffled(rng, roster)]),
              facts=Lst(_shuffled(rng, facts)),
              convention=Lst([Pred("records_complete_for", Ident("roster"))]),
              query=Pred("has", Ident(ent), Ident(prop)))
    return (obs, _shuffled(rng, ["yes", "no", "unknown"]), case,
            {"roster": list(roster), "queried": [ent, prop], "case": case,
             "properties": {e: list(v) for e, v in has.items()}})


class OpenWorldLanguage(Lesson):
    """False vs unknown under a closed roster."""

    id = "open_world_language"
    level = 56
    tags = ("analogy", "causality", "planning", "programs")
    teaches = "false vs unknown under a closed roster"
    capabilities = ('metareasoning', 'belief_modeling')
    axes = {'reasoning_depth': 3, 'world_complexity': 2, 'ambiguity': 3}
    answers = ['yes', 'no', 'unknown']

    generate = staticmethod(gen_open_world_language)
