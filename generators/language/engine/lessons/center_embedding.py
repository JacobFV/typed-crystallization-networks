"""Supplementary lesson: ``center_embedding`` — verb of the outermost subject under nesting.

Supplementary syntax and semantics.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec, Tok
from ..lesson import Lesson
from ..generators.base import NAMES
from ..generators.extra import _shuffled, adverbs, verbs


def gen_center_embedding(rng: random.Random, ctx):
    """``N1 N2 N3 V3 V2 V1``: the verb belonging to the *outermost* subject is
    the innermost-last one, so the pairing can only be recovered by unwinding
    the whole nesting. Depth varies and is recorded."""
    depth = rng.randint(*ctx.span((2, 4), (3, 6)))
    subs = rng.sample(NAMES, depth)
    chosen = rng.sample(verbs(), depth)
    toks = list(subs) + list(reversed(chosen))
    if rng.random() < 0.5:                       # a trailing adverb, so "last token" fails
        toks.append(rng.choice(adverbs()))
    # Always asking for the *outermost* subject fixes the answer at the last
    # verb of the sentence, which "take the option mentioned last" reads off
    # without unwinding anything.  Querying a subject drawn at random keeps the
    # pairing exactly as hard -- subject i still pairs with the verb at
    # ``depth + (depth - 1 - i)`` -- and moves the answer around.
    if not ctx.hardens("center_embedding"):
        obs = Rec(sentence=Lst([Tok(w) for w in toks]),
                  query=Pred("verb_of", Ident(subs[0])))
        return (obs, _shuffled(rng, chosen), chosen[0],
                {"depth": depth, "pairs": dict(zip(subs, chosen)), "length": len(toks)})
    q = rng.randrange(depth)
    obs = Rec(sentence=Lst([Tok(w) for w in toks]), query=Pred("verb_of", Ident(subs[q])))
    return (obs, _shuffled(rng, chosen), chosen[q],
            {"depth": depth, "pairs": dict(zip(subs, chosen)), "length": len(toks),
             "queried": subs[q]})


class CenterEmbedding(Lesson):
    """Verb of the outermost subject under nesting."""

    id = "center_embedding"
    level = 21
    tags = ("syntax", "semantics", "supplementary")
    teaches = "verb of the outermost subject under nesting"
    capabilities = ()
    axes = {'grammar_complexity': 4, 'recursion_depth': 4, 'compositional_depth': 3}

    generate = staticmethod(gen_center_embedding)
