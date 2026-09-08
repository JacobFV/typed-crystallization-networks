"""``narrative_modeling`` — the causally pivotal event, verified by deletion and re-simulation.

History, narrative, perspective, and identity.
"""

from __future__ import annotations

import random
from typing import Sequence

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.reflective import _labels, _nonces, _shuffled


def gen_narrative_modeling(rng: random.Random, ctx):
    """Which event was the pivot — the one whose removal changes the ending?

    Events fire only when their precondition already holds, so the story is a
    chain of enablements with redundant links in most places and exactly one
    bottleneck. Pivotality is defined operationally and verified by deleting
    each event in turn and re-running the whole narrative; worlds with zero or
    several pivots are discarded.
    """
    stages = ctx.at(3, 6, default=3)                       # links in the chain of enablements
    fallback = None
    for _ in range(400):
        flags = _nonces(rng, stages + 2, 3)
        chain, spare, unused = flags[:stages], flags[stages], flags[stages + 1]
        goal = chain[-1]
        singleton = rng.choice(list(range(1, stages + 1)))
        order: list[tuple[str, str]] = []                  # (precondition, effect)
        for stage in range(1, stages + 1):                 # one stage is the bottleneck
            link = ("none" if stage == 1 else chain[stage - 2], chain[stage - 1])
            order += [link] * (1 if singleton == stage else 2)
        noise = [(rng.choice([spare, unused, "none"]), rng.choice([spare, unused]))
                 for _ in range(rng.randint(1, 2))]
        seq: list[tuple[str, str]] = list(order)
        for nz in noise:
            seq.insert(rng.randrange(len(seq) + 1), nz)
        ids = _labels(rng, "e", len(seq))
        story = list(zip(ids, seq))

        def simulate(evs: Sequence[tuple[str, tuple[str, str]]]) -> bool:
            state = {"none"}
            for _, (pre, eff) in evs:
                if pre in state:
                    state.add(eff)
            return goal in state

        base = simulate(story)
        pivots = [i for i, _ in story
                  if simulate([e for e in story if e[0] != i]) != base]
        cand = (story, goal, pivots[0] if pivots else ids[0], ids, base)
        if fallback is None:
            fallback = cand
        if len(pivots) == 1 and base:
            fallback = cand
            break
    story, goal, answer, ids, base = fallback
    obs = Rec(events=Lst([Pred("event", Ident(i), Ident(pre), Ident(eff)) for i, (pre, eff) in story]),
              semantics=Pred("event_fires_only_if", Pred("precondition_already_holds")),
              outcome=Pred("holds_at_end", Ident(goal)),
              query=Pred("pivotal_for", Ident(goal)))
    return obs, _shuffled(rng, [i for i, _ in story]), answer, {"pivot": answer,
                                                                "n_events": len(story),
                                                                "outcome": bool(base)}


class NarrativeModeling(Lesson):
    """The causally pivotal event, verified by deletion and re-simulation."""

    id = "narrative_modeling"
    level = 132
    tags = ("history", "narrative", "perspective", "identity")
    teaches = "the causally pivotal event, verified by deletion and re-simulation"
    capabilities = ('causal_reasoning', 'temporal_reasoning', 'abstraction')
    axes = {'reasoning_depth': 5, 'discourse_horizon': 4, 'world_complexity': 4, 'compositional_depth': 3}

    generate = staticmethod(gen_narrative_modeling)
