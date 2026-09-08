"""``definitions`` — a novel word defined compositionally, then applied.

Language as action.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.base import COLORS, SHAPES
from ..generators.extra import SIZES, _nonce, _objects, _shuffled


def gen_definitions(rng: random.Random, ctx):
    """A novel word is *defined compositionally in the episode* and then applied:
    ``a dax is any red cube``; half the episodes stack a second definition on the
    first (``a blicket is any big dax``), so the word must be expanded before it
    can be used. Nothing carries between episodes."""
    nc, ns = ctx.at(2, 3, default=2), ctx.at(2, 3, default=2)   # the colour x shape grid
    objs = _objects(rng, nc * ns)
    cols = rng.sample(COLORS, nc)
    shapes = rng.sample(SHAPES, ns)
    for o, (c, s) in zip(objs, [(c, s) for c in cols for s in shapes]):
        o["color"], o["shape"] = c, s
    for o in objs:
        o["size"] = rng.choice(SIZES)

    word = _nonce(rng, rng.randint(3, 4))
    two_level = rng.random() < 0.5
    if two_level:
        # level 1 denotes two objects; level 2 narrows to exactly one by size
        color = rng.choice(cols)
        group = [o for o in objs if o["color"] == color]
        big = rng.choice(group)
        for o in group:
            o["size"] = "big" if o is big else "small"
        word2 = word
        while word2 == word:
            word2 = _nonce(rng, rng.randint(3, 4))
        defs = [Pred("define", Ident(word), Pred("color", Ident(color))),
                Pred("define", Ident(word2), Pred("and", Ident(word), Pred("size", Ident("big"))))]
        asked, target = word2, big
        hidden = {"levels": 2, "defs": {word: color, word2: f"{word}+big"}}
    else:
        target = rng.choice(objs)
        defs = [Pred("define", Ident(word),
                     Pred("and", Pred("color", Ident(target["color"])),
                          Pred("shape", Ident(target["shape"]))))]
        asked = word
        hidden = {"levels": 1, "defs": {word: f"{target['color']}+{target['shape']}"}}

    shown = _shuffled(rng, objs)
    scene = [Pred("obj", Ident(o["id"]), Ident(o["color"]), Ident(o["shape"])) for o in shown]
    scene += [Pred("size", Ident(o["id"]), Ident(o["size"])) for o in shown]
    obs = Rec(lexicon=Lst(_shuffled(rng, defs)), scene=Lst(scene),
              query=Pred("find", Ident(asked)))
    hidden["target"] = target["id"]
    return obs, _shuffled(rng, [o["id"] for o in objs]), target["id"], hidden


class Definitions(Lesson):
    """A novel word defined compositionally, then applied."""

    id = "definitions"
    level = 28
    tags = ("pragmatics", "language-as-action")
    teaches = "a novel word defined compositionally, then applied"
    capabilities = ()
    axes = {'lexical_novelty': 4, 'compositional_depth': 4, 'reasoning_depth': 3}

    generate = staticmethod(gen_definitions)
