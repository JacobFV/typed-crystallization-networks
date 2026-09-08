"""``lexicon_induction`` — acquiring a new language in-episode.

Language as action.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.base import COLORS, _scene


def gen_lexicon_induction(rng: random.Random, ctx):
    """Every episode invents a NEW language. Support examples ground novel words;
    the query uses them compositionally. Nothing carries over between episodes,
    so a learner that memorizes a vocabulary learns nothing here."""
    n_words = ctx.at(3, 5, default=3)            # words to ground, one scene object each
    words = ["".join(rng.choice("kmtszlp") for _ in range(3)) for _ in range(n_words)]
    colors = rng.sample(COLORS, n_words)
    lex = dict(zip(words, colors))
    support = [Pred("says", Ident(w), Ident(c)) for w, c in lex.items()]
    rng.shuffle(support)
    objs = _scene(rng, n_words + 1)
    # every colour distinct, so "find <word>" denotes exactly one object
    palette = colors + [c for c in COLORS if c not in colors][:1]
    for o, c in zip(objs, palette):
        o["color"] = c
    w = rng.choice(words)
    tgt = next(o for o in objs if o["color"] == lex[w])
    obs = Rec(lexicon=Lst(support),
              scene=Lst([Pred("obj", Ident(o["id"]), Ident(o["color"]), Ident(o["shape"]),
                              Num(o["x"]), Num(o["y"])) for o in objs]),
              query=Pred("find", Ident(w)))
    return obs, [o["id"] for o in objs], tgt["id"], {"lexicon": lex}


class LexiconInduction(Lesson):
    """Acquiring a new language in-episode."""

    id = "lexicon_induction"
    level = 14
    tags = ("pragmatics", "language-as-action")
    teaches = "acquiring a new language in-episode"
    capabilities = ()
    axes = {'lexical_novelty': 3, 'compositional_depth': 2, 'reasoning_depth': 2}

    generate = staticmethod(gen_lexicon_induction)
