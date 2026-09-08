"""``temporal_language`` — before / after / while / until over a trace.

Compositional semantics and logical language.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.semantics import EVENT_NAMES, _shuffled


def gen_temporal_language(rng: random.Random, ctx):
    """Interval traces with ``before / after / while / until``.

    The generator enumerates every (relation, anchor) pair whose denotation is a
    *singleton* under the interval semantics and samples one of those, so the
    ordering question always has exactly one correct event.
    """
    for _ in range(400):
        names = rng.sample(EVENT_NAMES, ctx.at(5, 8, default=5))
        evs = []
        for nm in names:
            s = rng.randint(0, 10)
            evs.append((nm, s, s + rng.choice([1, 1, 2, 3])))
        by_rel: dict[str, list[tuple[str, str]]] = {}
        for a in evs:
            rest = [e for e in evs if e[0] != a[0]]
            sets = {
                "before": [e for e in rest if e[2] <= a[1]],
                "after": [e for e in rest if e[1] >= a[2]],
                "while": [e for e in rest if e[1] < a[2] and a[1] < e[2]],
                "until": [e for e in rest if e[2] == a[1]],
            }
            for rel, s in sets.items():
                if len(s) == 1:
                    by_rel.setdefault(rel, []).append((a[0], s[0][0]))
        if by_rel:
            rel = rng.choice(sorted(by_rel))
            anchor, answer = rng.choice(by_rel[rel])
            break
    else:                                                  # pragma: no cover
        raise RuntimeError("no unique temporal relation found")

    trace = _shuffled(rng, [Pred("event", Ident(n), Num(s), Num(e)) for n, s, e in evs])
    obs = Rec(trace=Lst(trace),
              connectives=Lst([Ident(c) for c in ("before", "after", "while", "until")]),
              query=Pred("which_event", Ident(rel), Ident(anchor)))
    return obs, _shuffled(rng, names), answer, {"relation": rel, "anchor": anchor,
                                                "events": [list(e) for e in evs]}


class TemporalLanguage(Lesson):
    """Before / after / while / until over a trace."""

    id = "temporal_language"
    level = 17
    tags = ("compositional-semantics", "logic")
    teaches = "before / after / while / until over a trace"
    capabilities = ('temporal_reasoning', 'sequence_memory')
    axes = {'reasoning_depth': 3, 'discourse_horizon': 3, 'world_complexity': 3}

    generate = staticmethod(gen_temporal_language)
