"""``finite_state_language`` — stateful syntax / automata induction.

Symbols, grounding, and elementary language.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec, Str
from ..lesson import Lesson


def gen_finite_state(rng: random.Random, ctx):
    """Accept/reject strings of a hidden regular language (parity of a symbol)."""
    target = rng.choice("ab")
    parity = rng.choice([0, 1])
    s = [rng.choice("ab") for _ in range(rng.randint(*ctx.span((2, 8), (6, 16))))]
    ok = (s.count(target) % 2) == parity
    shown = []
    for _ in range(4):
        t = [rng.choice("ab") for _ in range(rng.randint(*ctx.span((2, 6), (5, 12))))]
        shown.append((t, (t.count(target) % 2) == parity))
    obs = Rec(examples=Lst([Pred("ex", Str("".join(t)), Ident("yes" if v else "no")) for t, v in shown]),
              query=Pred("accept", Str("".join(s))))
    return obs, ["yes", "no"], ("yes" if ok else "no"), {"symbol": target, "parity": parity}


class FiniteStateLanguage(Lesson):
    """Stateful syntax / automata induction."""

    id = "finite_state_language"
    level = 6
    tags = ("symbols", "grounding", "elementary-language")
    teaches = "stateful syntax / automata induction"
    capabilities = ()
    axes = {'grammar_complexity': 2, 'reasoning_depth': 2}
    answers = ['yes', 'no']

    generate = staticmethod(gen_finite_state)
