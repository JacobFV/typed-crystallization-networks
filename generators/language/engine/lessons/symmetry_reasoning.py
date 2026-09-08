"""``symmetry_reasoning`` — infer an unseen state from a symmetry group.

Scientific induction and model discovery.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.base import COLORS
from ..generators.science import _shuffled


def gen_symmetry_reasoning(rng: random.Random, ctx):
    """Use a symmetry to infer a state you were never shown.

    The world is declared invariant under a stated permutation, so colour is
    constant on each orbit of that permutation; one site is hidden and its colour
    is fixed by any other site in its orbit. This is symmetry used as an
    inference rule rather than recognized as a pattern. The hidden site never
    sits in the largest orbit, so answering "the commonest visible colour" is
    wrong more often than not.
    """
    for _ in range(400):
        n = rng.choice(ctx.among([[6, 7, 8], [8, 9, 10], [10, 11, 12]]))
        perm = _shuffled(rng, range(n))
        orbits: list[list[int]] = []
        seen: set[int] = set()
        for i in range(n):
            if i in seen:
                continue
            orb, j = [], i
            while j not in seen:
                seen.add(j)
                orb.append(j)
                j = perm[j]
            orbits.append(orb)
        if len(orbits) < 2 or len(COLORS) < len(orbits):
            continue
        sizes = [len(o) for o in orbits]
        big = max(sizes)
        choices = [i for i, o in enumerate(orbits) if 2 <= len(o) < big]
        if not choices:
            continue                                       # need a sibling *and* a bigger orbit
        palette = _shuffled(rng, COLORS)[: len(orbits)]
        color = {}
        for orb, c in zip(orbits, palette):
            for i in orb:
                color[i] = c
        oi = rng.choice(choices)
        masked = rng.choice(orbits[oi])
        obs = Rec(symmetry=Pred("invariant_under",
                                Lst([Num(perm[i]) for i in range(n)])),
                  state=Lst([Pred("site", Num(i), Ident(color[i])) for i in range(n) if i != masked]),
                  query=Pred("color_at", Num(masked)))
        hidden = {"orbits": [list(o) for o in orbits], "masked": masked,
                  "orbit_size": len(orbits[oi]), "answer": color[masked]}
        return obs, _shuffled(rng, COLORS), color[masked], hidden
    raise RuntimeError("symmetry_reasoning: no admissible episode")


class SymmetryReasoning(Lesson):
    """Infer an unseen state from a symmetry group."""

    id = "symmetry_reasoning"
    level = 78
    tags = ("science", "induction", "model-discovery")
    teaches = "infer an unseen state from a symmetry group"
    capabilities = ('abstraction', 'spatial_reasoning', 'scientific_induction')
    axes = {'reasoning_depth': 4, 'compositional_depth': 3, 'world_complexity': 3}
    answers = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']

    generate = staticmethod(gen_symmetry_reasoning)
