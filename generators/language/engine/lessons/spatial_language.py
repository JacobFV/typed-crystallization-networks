"""``spatial_language`` — egocentric and allocentric frames.

Compositional semantics and logical language.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.base import COLORS
from ..generators.semantics import DIRS, DIRVEC, EGO, _shuffled


def gen_spatial_language(rng: random.Random):
    """Egocentric and allocentric frames in the same lesson.

    Four objects sit one per compass direction from the agent, so every
    allocentric direction denotes exactly one object; the egocentric episodes
    ask for ``left/right/front/behind``, which denote only once the agent's
    ``facing`` is composed with the frame. The ``frame`` field makes which of
    the two is in force recoverable from the observation.
    """
    ids = _shuffled(rng, ["o0", "o1", "o2", "o3"])
    ax, ay = rng.randint(3, 6), rng.randint(3, 6)
    dir_of = dict(zip(ids, _shuffled(rng, DIRS)))
    scene = []
    for oid in ids:
        dx, dy = DIRVEC[dir_of[oid]]
        k = rng.randint(1, 3)
        scene.append(Pred("obj", Ident(oid), Ident(rng.choice(COLORS)),
                          Num(ax + dx * k), Num(ay + dy * k)))
    rng.shuffle(scene)

    facing = rng.choice(DIRS)
    frame = rng.choice(["egocentric", "allocentric"])
    if frame == "allocentric":
        rel = rng.choice(DIRS)
        want = rel
    else:
        rel = rng.choice(EGO)
        want = DIRS[(DIRS.index(facing) + EGO.index(rel)) % 4]
    answer = next(o for o in ids if dir_of[o] == want)

    obs = Rec(agent=Pred("agent", Num(ax), Num(ay), Ident(facing)),
              frame=Ident(frame),
              scene=Lst(scene),
              query=Pred("which_object", Ident(rel)))
    return obs, _shuffled(rng, ids), answer, {"frame": frame, "facing": facing,
                                              "relation": rel, "resolved_direction": want}


class SpatialLanguage(Lesson):
    """Egocentric and allocentric frames."""

    id = "spatial_language"
    level = 16
    tags = ("compositional-semantics", "logic")
    teaches = "egocentric and allocentric frames"
    capabilities = ('spatial_reasoning', 'lexical_grounding')
    axes = {'world_complexity': 3, 'compositional_depth': 3, 'ambiguity': 2}

    generate = staticmethod(gen_spatial_language)
