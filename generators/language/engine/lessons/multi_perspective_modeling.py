"""``multi_perspective_modeling`` — what an agent believes, given what it was there to see.

History, narrative, perspective, and identity.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.base import NAMES
from ..generators.reflective import _shuffled


def gen_multi_perspective_modeling(rng: random.Random, ctx):
    """Where does *this* agent think the ball is?

    Agents come and go while the ball is moved; an agent's belief is whatever it
    last saw, which is the truth only if it was in the room for the final move.
    Half the episodes are built so the queried agent's belief and the world
    disagree, so an agent that reports the true location is right at chance.
    """
    boxes = [f"box{i}" for i in range(1, 5)]
    want_differ = rng.random() < 0.55
    fallback = None
    for _ in range(300):
        agents = rng.sample(NAMES, rng.randint(2, 3))
        loc = rng.choice(boxes)
        timeline: list[tuple[int, str, str]] = [(0, "place", loc)]
        present = {a: True for a in agents}
        belief = {a: loc for a in agents}
        t = 1
        for _ in range(rng.randint(*ctx.span((3, 5), (9, 15)))):
            roll = rng.random()
            if roll < 0.45:
                nxt = rng.choice([b for b in boxes if b != loc])
                loc = nxt
                timeline.append((t, "move", nxt))
                for a in agents:
                    if present[a]:
                        belief[a] = nxt
            else:
                a = rng.choice(agents)
                if present[a]:
                    timeline.append((t, "exit", a))
                    present[a] = False
                else:
                    # re-entering reveals nothing: the belief the agent left with stands
                    timeline.append((t, "enter", a))
                    present[a] = True
            t += 1
        who = rng.choice(agents)
        cand = (agents, timeline, who, belief[who], loc)
        if fallback is None:
            fallback = cand
        if (belief[who] != loc) == want_differ:
            fallback = cand
            break
    agents, timeline, who, answer, loc = fallback
    syms = []
    for t, kind, arg in timeline:
        syms.append(Pred(kind, Num(t), Ident(arg)))
    obs = Rec(agents=Lst([Ident(a) for a in agents]),
              timeline=Lst(syms),
              semantics=Pred("an_agent_sees_a_move", Pred("only_while_present")),
              query=Pred("believes_ball_in", Ident(who)))
    return obs, _shuffled(rng, boxes), answer, {"truth": loc, "belief": answer,
                                                "agent": who, "differs": answer != loc}


class MultiPerspectiveModeling(Lesson):
    """What an agent believes, given what it was there to see."""

    id = "multi_perspective_modeling"
    level = 133
    tags = ("history", "narrative", "perspective", "identity")
    teaches = "what an agent believes, given what it was there to see"
    capabilities = ('belief_modeling', 'temporal_reasoning', 'multi_agent_coordination')
    axes = {'reasoning_depth': 4, 'discourse_horizon': 4, 'world_complexity': 3, 'ambiguity': 3}

    generate = staticmethod(gen_multi_perspective_modeling)
