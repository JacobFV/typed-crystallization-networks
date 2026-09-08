"""``event_semantics`` — participants and times of named events.

Compositional semantics and logical language.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.base import NAMES
from ..generators.extra import _shuffled, verbs


def gen_event_semantics(rng: random.Random, ctx):
    """Neo-Davidsonian events: each has an id, a predicate, two participants and
    a time. Queries ask either for a participant of a named event or for the
    temporal extremum. Times are a shuffled sample, so the event id says nothing
    about when it happened, and the event list order says nothing either."""
    # actors and themes are disjoint draws from the same six names, so three
    # events is the most this world can hold; the knob raises the floor to it
    n = rng.randint(*ctx.span((2, 3), (3, 3)))
    actors = rng.sample(NAMES, n)
    themes = rng.sample([x for x in NAMES if x not in actors], n)
    chosen = rng.sample(verbs(), n)
    times = rng.sample(range(1, 12), n)
    events = [{"id": f"e{i}", "verb": chosen[i], "actor": actors[i],
               "theme": themes[i], "time": times[i]} for i in range(n)]
    shown = _shuffled(rng, events)
    facts = Lst([Pred("event", Ident(e["id"]), Ident(e["verb"]), Ident(e["actor"]),
                      Ident(e["theme"]), Num(e["time"])) for e in shown])

    kind = rng.choice(["actor_of", "theme_of", "order"])
    if kind == "order":
        extremum = rng.choice(["earliest", "latest"])
        pick = min if extremum == "earliest" else max
        target = pick(events, key=lambda e: e["time"])
        obs = Rec(events=facts, query=Ident(extremum))
        answers = _shuffled(rng, [e["id"] for e in events])
        return obs, answers, target["id"], {"kind": extremum, "n_events": n,
                                            "times": {e["id"]: e["time"] for e in events}}
    ev = rng.choice(events)
    obs = Rec(events=facts, query=Pred(kind, Ident(ev["id"])))
    answers = _shuffled(rng, actors + themes)
    answer = ev["actor"] if kind == "actor_of" else ev["theme"]
    return obs, answers, answer, {"kind": kind, "n_events": n, "event": ev["id"]}


class EventSemantics(Lesson):
    """Participants and times of named events."""

    id = "event_semantics"
    level = 27
    tags = ("compositional-semantics", "logic")
    teaches = "participants and times of named events"
    capabilities = ()
    axes = {'world_complexity': 3, 'discourse_horizon': 3, 'compositional_depth': 3}

    generate = staticmethod(gen_event_semantics)
