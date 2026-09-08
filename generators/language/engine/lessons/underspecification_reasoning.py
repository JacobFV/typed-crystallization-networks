"""``underspecification_reasoning`` — count what the instruction actually fixes.

Open-ended epistemology.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.base import COLORS
from ..generators.selfmodel import _labels, _rules, _shuffled


def gen_underspecification_reasoning(rng: random.Random, ctx):
    """Separate what the instruction fixes from what it leaves open.

    The instruction constrains the *types* of its two arguments and a stated
    compatibility relation constrains their colours; everything else is
    conventional, not determined. The answer is the number of fully specified
    actions consistent with what was actually said.
    """
    k = rng.choice([1, 2, 3, 4, 5])
    cube_ids = _labels(rng, "c", 3)
    box_ids = _labels(rng, "b", 3)
    cube_cols = rng.sample(COLORS, 3)
    box_cols = rng.sample([c for c in COLORS if c not in cube_cols], 3)
    pairs = [(i, j) for i in range(3) for j in range(3)]
    chosen = rng.sample(pairs, k)
    fits = [(cube_cols[i], box_cols[j]) for i, j in chosen]
    # distractor compatibilities over colours that no object in the scene has
    spare = [c for c in COLORS if c not in cube_cols and c not in box_cols]
    for c in spare:
        fits.append((c, rng.choice(box_cols)))
    n = sum(1 for i in range(3) for j in range(3) if (cube_cols[i], box_cols[j]) in fits)
    assert n == k

    scene = [Pred("obj", Ident(cube_ids[i]), Ident("cube"), Ident(cube_cols[i])) for i in range(3)]
    scene += [Pred("obj", Ident(box_ids[j]), Ident("box"), Ident(box_cols[j])) for j in range(3)]
    # decoy objects: a cube wearing a box colour (and vice versa) fits nothing, so
    # the count is unchanged and only the number of pairs to check goes up
    for e in range(ctx.at(0, 4, default=0)):
        scene.append(Pred("obj", Ident(f"c{3 + e}"), Ident("cube"), Ident(rng.choice(box_cols))))
        scene.append(Pred("obj", Ident(f"b{3 + e}"), Ident("box"), Ident(rng.choice(cube_cols))))
    obs = Rec(scene=Lst(_shuffled(rng, scene)),
              compatibility=Lst(_shuffled(rng, [Pred("fits", Ident(a), Ident(b)) for a, b in fits])),
              instruction=Lst([Pred("put", Ident("X"), Ident("Y")),
                               Pred("typed", Ident("X"), Ident("cube")),
                               Pred("typed", Ident("Y"), Ident("box"))]),
              rules=_rules("an_instantiation_assigns_one_scene_object_to_each_of_x_and_y_respecting_its_type",
                           "an_instantiation_is_admissible_iff_fits_holds_of_the_two_objects_colours"),
              query=Ident("count_admissible_instantiations"))
    return (obs, _shuffled(rng, [1, 2, 3, 4, 5]), k,
            {"admissible": k, "cube_colors": cube_cols, "box_colors": box_cols})


class UnderspecificationReasoning(Lesson):
    """Count what the instruction actually fixes."""

    id = "underspecification_reasoning"
    level = 156
    tags = ("open-ended-epistemology",)
    teaches = "count what the instruction actually fixes"
    capabilities = ('ambiguity_management', 'quantification')
    axes = {'ambiguity': 4, 'reasoning_depth': 3, 'compositional_depth': 3}
    answers = [1, 2, 3, 4, 5]

    generate = staticmethod(gen_underspecification_reasoning)
