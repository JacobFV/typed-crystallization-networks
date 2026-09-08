"""``world_model_synthesis`` — query a world model of containment and events.

Open-ended epistemology.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.selfmodel import _labels, _rules, _shuffled


def gen_world_model_synthesis(rng: random.Random, ctx):
    """A world model with containment, nesting, and timed processes; query it.

    Objects sit inside crates or inside other objects, crates sit in rooms, and
    crates are moved at known times. Answering "where is this at time t" means
    following the containment chain *and* replaying the events in order — either
    half alone gives the wrong room.
    """
    rooms = _labels(rng, "room", 4)
    crates = _labels(rng, "crate", 3)
    objs = _labels(rng, "obj", ctx.at(4, 9, default=4))
    # objs[0] nests inside objs[1], which may nest inside objs[2], and so on;
    # the rest sit directly in a crate. The nesting depth is the knob: the
    # containment chain to follow before the events can be replayed at all.
    depth = min(len(objs) - 1, ctx.at(1, 5, default=1))
    holder = {objs[i]: ("obj", objs[i + 1]) for i in range(depth)}
    for o in objs[depth:]:
        holder[o] = ("crate", rng.choice(crates))
    at0 = {c: rng.choice(rooms) for c in crates}
    events = [(t, rng.choice(crates), rng.choice(rooms)) for t in (1, 2)]
    t_query = rng.choice([0, 1, 2])
    obj = rng.choice(objs)

    at = dict(at0)
    for t, c, r in events:
        if t <= t_query:
            at[c] = r
    cur = obj
    while holder[cur][0] == "obj":
        cur = holder[cur][1]
    answer = at[holder[cur][1]]

    facts = [Pred("inside", Ident(o), Ident(holder[o][1])) for o in objs]
    facts += [Pred("stands_in", Ident(c), Ident(at0[c])) for c in crates]
    facts += [Pred("moved", Num(t), Ident(c), Ident(r)) for t, c, r in events]
    obs = Rec(world=Lst(_shuffled(rng, facts)),
              rules=_rules("stands_in_gives_the_room_of_a_crate_at_time_0",
                           "moved_t_c_r_puts_crate_c_in_room_r_from_time_t_onward",
                           "an_object_is_in_the_room_of_the_crate_that_transitively_contains_it"),
              query=Pred("room_of", Ident(obj), Num(t_query)))
    return (obs, _shuffled(rng, rooms), answer,
            {"object": obj, "time": t_query, "chain": cur, "events": [list(e) for e in events]})


class WorldModelSynthesis(Lesson):
    """Query a world model of containment and events."""

    id = "world_model_synthesis"
    level = 150
    tags = ("open-ended-epistemology",)
    teaches = "query a world model of containment and events"
    capabilities = ('ontology_learning', 'temporal_reasoning', 'causal_reasoning')
    axes = {'reasoning_depth': 4, 'world_complexity': 4, 'discourse_horizon': 3}

    generate = staticmethod(gen_world_model_synthesis)
