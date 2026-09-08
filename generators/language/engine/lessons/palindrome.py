"""Supplementary lesson: ``palindrome`` — mirror symmetry over a whole string.

Supplementary syntax and semantics.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Rec, Tok
from ..lesson import Lesson
from ..generators.extra import _yesno


def gen_palindrome(rng: random.Random, ctx):
    """Is this symbol string a palindrome? Negatives are one edit away, so the
    only thing that separates the classes is the mirror relation itself."""
    n = rng.randint(*ctx.span((3, 9), (7, 15)))
    alpha = list("abcdef")
    half = [rng.choice(alpha) for _ in range((n + 1) // 2)]
    s = half + half[: n // 2][::-1]
    if rng.random() < 0.5:                       # hard negative: perturb one slot
        for _ in range(24):
            i = rng.randrange(n)
            t = list(s)
            t[i] = rng.choice([c for c in alpha if c != s[i]])
            if t != t[::-1]:
                s = t
                break
    truth = s == s[::-1]
    obs = Rec(symbols=Lst([Tok(c) for c in s]), query=Ident("palindrome"))
    answers, answer = _yesno(rng, truth)
    return obs, answers, answer, {"length": n, "string": "".join(s), "palindrome": truth}


class Palindrome(Lesson):
    """Mirror symmetry over a whole string."""

    id = "palindrome"
    level = 15
    tags = ("syntax", "semantics", "supplementary")
    teaches = "mirror symmetry over a whole string"
    capabilities = ()
    axes = {'recursion_depth': 3, 'grammar_complexity': 2, 'reasoning_depth': 2}
    answers = ['yes', 'no']

    generate = staticmethod(gen_palindrome)
