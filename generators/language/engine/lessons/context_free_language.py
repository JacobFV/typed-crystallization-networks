"""``context_free_language`` — recursion and stack-like state.

Symbols, grounding, and elementary language.
"""

from __future__ import annotations

import random

from .._structure import Ident, Rec, Str
from ..lesson import Lesson
from ..generators.base import _balanced, _is_balanced


def gen_context_free(rng: random.Random, ctx):
    """Balanced brackets: the canonical test that a learner has stack-like state."""
    depth = rng.randint(*ctx.span((1, 4), (3, 7)))
    balanced = rng.random() < 0.5
    if balanced:
        s = _balanced(rng, depth)
    else:
        s = list(_balanced(rng, depth))
        i = rng.randrange(len(s))
        s[i] = "(" if s[i] == ")" else ")"
        s = "".join(s)
    obs = Rec(string=Str(s), query=Ident("balanced"))
    ok = _is_balanced(s)
    return obs, ["yes", "no"], ("yes" if ok else "no"), {"depth": depth, "string": s}


class ContextFreeLanguage(Lesson):
    """Recursion and stack-like state."""

    id = "context_free_language"
    level = 7
    tags = ("symbols", "grounding", "elementary-language")
    teaches = "recursion and stack-like state"
    capabilities = ()
    axes = {'grammar_complexity': 3, 'recursion_depth': 3}
    answers = ['yes', 'no']

    generate = staticmethod(gen_context_free)
