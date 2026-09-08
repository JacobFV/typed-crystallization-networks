"""``symbol_equivalence`` — many-to-one lexical semantics.

Symbols, grounding, and elementary language.
"""

from __future__ import annotations

import random
import string

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.base import COLORS, _scene


def gen_symbol_equivalence(rng: random.Random, ctx):
    """Two symbols denote the same thing; the alias is defined *in the episode*."""
    objs = _scene(rng, rng.randint(*ctx.span((3, 4), (6, 10))))
    tgt = rng.choice(objs)
    alias = "".join(rng.choice(string.ascii_lowercase) for _ in range(4))
    facts = [Pred("means", Ident(alias), Ident(tgt["color"]))]
    obs = Rec(scene=Lst([Pred("obj", Ident(o["id"]), Ident(o["color"]), Ident(o["shape"]),
                             Num(o["x"]), Num(o["y"])) for o in objs]),
              lexicon=Lst(facts), query=Pred("which_color", Ident(alias)))
    return obs, COLORS, tgt["color"], {"alias": alias, "denotes": tgt["color"]}


class SymbolEquivalence(Lesson):
    """Many-to-one lexical semantics."""

    id = "symbol_equivalence"
    level = 2
    tags = ("symbols", "grounding", "elementary-language")
    teaches = "many-to-one lexical semantics"
    capabilities = ()
    axes = {'lexical_novelty': 2, 'compositional_depth': 1}
    answers = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']

    generate = staticmethod(gen_symbol_equivalence)
