"""``entailment`` — entailed / contradicted / unknown from premises.

Language as action.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.base import COLORS, SHAPES
from ..generators.extra import _shuffled


def gen_entailment(rng: random.Random, ctx):
    """Three-way natural-language inference over a tiny closed world: a universal
    rule, some ground facts, a uniqueness axiom for colour, and one entity about
    which *nothing* is stated. The label is chosen first and the premises are
    then built to realize it, with the roles assigned to shuffled entity names so
    no name predicts the label."""
    e1, e2, e3 = rng.sample([f"o{i}" for i in range(6)], 3)
    shape_x, shape_z = rng.sample(SHAPES, 2)
    color_y, color_w = rng.sample(COLORS, 2)
    premises = [
        Pred("all_are", Ident(shape_x), Ident(color_y)),   # every X-shaped thing is Y
        Pred("shape", Ident(e1), Ident(shape_x)),
        Pred("shape", Ident(e2), Ident(shape_z)),
        Pred("color", Ident(e2), Ident(color_w)),
        Pred("object", Ident(e3)),                         # declared, nothing known
        Pred("axiom", Pred("one_color_per_object")),
    ]
    # further rules and entities that bear on nothing the query asks about: the
    # shapes are ones no queried entity has, so the three labels are untouched
    spare_shapes = [s for s in SHAPES if s not in (shape_x, shape_z)]
    spare_colors = [c for c in COLORS if c not in (color_y, color_w)]
    for i in range(ctx.at(0, 4, default=0)):
        premises.append(Pred("all_are", Ident(spare_shapes[i]), Ident(spare_colors[i])))
        premises.append(Pred("shape", Ident(f"o{6 + i}"), Ident(spare_shapes[i])))
    label = rng.choice(["entailed", "contradicted", "unknown"])
    if label == "entailed":
        if rng.random() < 0.6:
            ent, col, form = e1, color_y, "rule"           # X-shaped, so Y
        else:
            ent, col, form = e2, color_w, "fact"           # stated directly
    elif label == "contradicted":
        if rng.random() < 0.6:
            ent = e1
            col = rng.choice([c for c in COLORS if c != color_y])
            form = "rule"
        else:
            ent, col, form = e2, color_y, "fact"           # e2 is W, colours unique
    else:
        ent, col, form = e3, rng.choice(COLORS), "silent"
    obs = Rec(premises=Lst(_shuffled(rng, premises)),
              query=Pred("conclusion", Pred("color", Ident(ent), Ident(col))))
    return (obs, _shuffled(rng, ["entailed", "contradicted", "unknown"]), label,
            {"label": label, "form": form, "rule": [shape_x, color_y],
             "conclusion": [ent, col], "silent_entity": e3})


class Entailment(Lesson):
    """Entailed / contradicted / unknown from premises."""

    id = "entailment"
    level = 30
    tags = ("pragmatics", "language-as-action")
    teaches = "entailed / contradicted / unknown from premises"
    capabilities = ()
    axes = {'reasoning_depth': 4, 'compositional_depth': 3, 'ambiguity': 2}
    answers = ['entailed', 'contradicted', 'unknown']

    generate = staticmethod(gen_entailment)
