"""``symbolic_generalist`` — held-out mixture of other lessons, none appearing verbatim.

Ultimate transfer and open-world capstones.
"""

from __future__ import annotations

import random

from .._structure import Ident, Pred, Rec
from ..lesson import Lesson
from ..generators.capstone import _OWN_IDS, _dedup, _shuffled


_QUARTERS = ("north", "south", "east", "west", "centre")


def gen_symbolic_generalist(rng: random.Random, ctx):
    """A held-out mixture: two *different* lessons from the registry are
    instantiated into one composite world, and the query names which of the two
    worlds it is about.

    The mixture is drawn at call time from the registry, minus this section's
    own lessons (which would recurse), so it automatically covers every lesson
    any other section contributes. No episode is verbatim a curriculum episode —
    the learner must first work out which half of the record the question is
    about, in a vocabulary that is the union of both halves' answers — while the
    ground truth stays the component lesson's exact answer.
    """
    from ..registry import all_lessons

    own = _OWN_IDS
    pool = sorted((l for l in all_lessons().values()
                   if l.status == "implemented" and l.id not in own), key=lambda l: l.id)
    if len(pool) < 2:                                 # pragma: no cover - registry empty
        raise RuntimeError("symbolic_generalist needs at least two other implemented lessons")

    n_parts = min(len(_QUARTERS), len(pool), ctx.at(2, 5, default=2))
    picked = None
    for _ in range(20):
        drawn = rng.sample(pool, n_parts)
        try:
            parts = [l.invoke(rng) for l in drawn]
        except Exception:                             # a broken lesson must not break this one
            continue
        if any(ans not in list(voc) for _, voc, ans, _ in parts):
            continue
        picked = (drawn, [(o, list(v), a, h) for o, v, a, h in parts])
        break
    if picked is None:                                # pragma: no cover
        raise RuntimeError("symbolic_generalist could not draw a usable mixture")
    drawn, parts = picked

    first = rng.random() < 0.5
    slot_of = list(range(n_parts)) if first else list(reversed(range(n_parts)))
    placed = [parts[i] for i in slot_of]              # placed[j] occupies quarter j
    quarters = list(_QUARTERS[:n_parts])
    asked = rng.choice(quarters)
    j = quarters.index(asked)
    answer = placed[j][2]
    source = drawn[slot_of[j]].id
    obs = Rec(**{q: placed[j2][0] for j2, q in enumerate(quarters)},
              query=Pred("resolve_query_in", Ident(asked)))
    vocab = _shuffled(rng, _dedup([v for _, voc, _, _ in parts for v in voc]))
    hidden = {"components": [l.id for l in drawn], "asked": asked, "source": source,
              "answer": answer, "levels": [l.level for l in drawn]}
    return obs, vocab, answer, hidden


class SymbolicGeneralist(Lesson):
    """Held-out mixture of other lessons, none appearing verbatim."""

    id = "symbolic_generalist"
    level = 169
    tags = ("transfer", "capstone", "open-world")
    teaches = "held-out mixture of other lessons, none appearing verbatim"
    capabilities = ('abstraction', 'metareasoning', 'open_ended_discovery')
    axes = {'lexical_novelty': 4, 'compositional_depth': 4, 'reasoning_depth': 4, 'world_complexity': 4, 'ambiguity': 3}

    generate = staticmethod(gen_symbolic_generalist)
