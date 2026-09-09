"""``symbol_equivalence`` — many-to-one lexical semantics.

Symbols, grounding, and elementary language.
"""

from __future__ import annotations

import random
import string

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.base import COLORS, _scene
from ..generators.extra import _shuffled


def gen_symbol_equivalence(rng: random.Random, ctx):
    """Two symbols denote the same thing; the alias is defined *in the episode*."""
    objs = _scene(rng, rng.randint(*ctx.span((3, 4), (6, 10))))
    tgt = rng.choice(objs)
    alias = "".join(rng.choice(string.ascii_lowercase) for _ in range(4))
    facts = [Pred("means", Ident(alias), Ident(tgt["color"]))]
    denotes = tgt["color"]
    if ctx.hardens("symbol_equivalence"):
        # One alias means the answer colour is named twice -- once in the scene
        # and once in the lexicon -- and is the last colour printed, so "the
        # option that occurs most" and "the option that occurs last" are both
        # exact.  A lexicon of several aliases forces the queried one to be
        # looked up; the distractor aliases denote other colours in the scene,
        # so every colour present is named the same number of times.
        # Aliasing only the scene's colours leaves "a colour that the lexicon
        # names" as a two- or three-way narrowing of a six-way answer.  Every
        # colour gets an alias -- some get two, which is what "many-to-one"
        # means -- so the lexicon narrows nothing and the queried alias has to
        # be matched.
        pool = list(COLORS) + [rng.choice(list(COLORS)) for _ in range(2)]
        aliases = {}
        for c in pool:
            while True:
                a = "".join(rng.choice(string.ascii_lowercase) for _ in range(4))
                if a not in aliases:
                    aliases[a] = c
                    break
        # Deriving the query from a randomly chosen *object* makes the answer
        # proportional to how many objects wear each colour, so "the colour
        # named most often" is a real prior.  Choosing among the aliases makes
        # every aliased colour equally likely.
        alias = rng.choice(sorted(aliases))
        denotes = aliases[alias]
        facts = _shuffled(rng, [Pred("means", Ident(a), Ident(c))
                                for a, c in aliases.items()])
    obs = Rec(scene=Lst([Pred("obj", Ident(o["id"]), Ident(o["color"]), Ident(o["shape"]),
                             Num(o["x"]), Num(o["y"])) for o in objs]),
              lexicon=Lst(facts), query=Pred("which_color", Ident(alias)))
    return obs, COLORS, denotes, {"alias": alias, "denotes": denotes}


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
