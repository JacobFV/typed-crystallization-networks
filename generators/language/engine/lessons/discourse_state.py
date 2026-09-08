"""``discourse_state`` — salience as state across turns.

Language as action.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.semantics import THINGS, _shuffled


def gen_discourse_state(rng: random.Random, ctx):
    """Salience as state: entities are mentioned and *closed* over turns.

    The referent of "the animal we were talking about" is the most recently
    mentioned entity of that kind whose mention has not since been closed — so
    the answer depends on the whole turn sequence, not on the last line.
    """
    n_ents = ctx.at(4, 8, default=4)
    for _ in range(200):
        kind = rng.choice(sorted({k for _, k in THINGS}))
        same = [n for n, k in THINGS if k == kind]
        others = [n for n, k in THINGS if k != kind]
        n_same = rng.randint(2, min(3, len(same)))
        ents = rng.sample(same, n_same) + rng.sample(others, n_ents - n_same)
        ents = _shuffled(rng, ents)
        kind_of = {n: k for n, k in THINGS}

        turns: list[tuple[str, str]] = []
        for e in rng.sample([x for x in ents if kind_of[x] == kind], n_same):
            turns.append(("mention", e))            # every candidate is mentioned
        for _ in range(rng.randint(*ctx.span((1, 3), (6, 10)))):
            turns.append(("mention", rng.choice(ents)))
        rng.shuffle(turns)
        if rng.random() < 0.5:
            i = rng.randrange(len(turns) + 1)
            turns.insert(i, ("close", rng.choice(ents)))

        last: dict[str, int] = {}
        for t, (op, e) in enumerate(turns):
            last[e] = t if op == "mention" else -1
        active = [e for e in ents if kind_of[e] == kind and last.get(e, -1) >= 0]
        if not active:
            continue
        answer = max(active, key=lambda e: last[e])
        if sum(1 for e in active if last[e] == last[answer]) != 1:
            continue
        break
    else:                                                  # pragma: no cover
        raise RuntimeError("no resolvable discourse state")

    obs = Rec(entities=Lst([Pred("entity", Ident(e), Ident(kind_of[e])) for e in ents]),
              turns=Lst([Pred("turn", Num(t), Ident(op), Ident(e))
                         for t, (op, e) in enumerate(turns)]),
              query=Pred("most_salient", Ident(kind)))
    return obs, _shuffled(rng, ents), answer, {"kind": kind, "turns": [list(t) for t in turns]}


class DiscourseState(Lesson):
    """Salience as state across turns."""

    id = "discourse_state"
    level = 23
    tags = ("pragmatics", "language-as-action")
    teaches = "salience as state across turns"
    capabilities = ('sequence_memory', 'variable_binding')
    axes = {'discourse_horizon': 4, 'ambiguity': 2, 'world_complexity': 2}

    generate = staticmethod(gen_discourse_state)
