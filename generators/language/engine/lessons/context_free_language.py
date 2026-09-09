"""``context_free_language`` — recursion and stack-like state.

Symbols, grounding, and elementary language.
"""

from __future__ import annotations

import random

from .._structure import Ident, Rec, Str
from ..lesson import Lesson
from ..generators.base import _balanced, _is_balanced


def _count_preserving_negative(rng: random.Random, s: str) -> str | None:
    """A rearrangement of ``s`` with the same characters that is *not* balanced.

    A single-character flip always changes the number of brackets, so a learner
    that only counts them separates the classes perfectly and never needs a
    stack -- measured at 20,000 of 20,000 seeds in
    ``research/language-capability/RESULTS.md``.  A permutation leaves every
    count identical, so ``#( == #)`` is true of both classes and only the
    matching relation tells them apart.

    The first and last characters are held fixed where that is possible, so
    "starts with a closing bracket" does not become the next shortcut.
    """
    n = len(s)
    if n < 4:
        return None
    for keep_last in (True, False):                # first tier keeps both ends
        body = list(s[1:-1] if keep_last else s[1:])
        tail = s[-1] if keep_last else ""
        for _ in range(64):
            rng.shuffle(body)
            t = s[0] + "".join(body) + tail
            if not _is_balanced(t):
                return t
    return None


def gen_context_free(rng: random.Random, ctx):
    """Balanced brackets: the canonical test that a learner has stack-like state."""
    depth = rng.randint(*ctx.span((1, 4), (3, 7)))
    balanced = rng.random() < 0.5
    if ctx.hardens("context_free_language"):
        # Both classes are drawn from the same string first, so length and
        # bracket counts are identical in distribution across the two answers.
        for _ in range(16):
            base = _balanced(rng, depth)
            if len(base) >= 4:
                break
        if balanced:
            s = base
        else:
            t = _count_preserving_negative(rng, base)
            if t is None:
                b = list(base)
                i = rng.randrange(len(b))
                b[i] = "(" if b[i] == ")" else ")"
                t = "".join(b)
            s = t
    elif balanced:
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
