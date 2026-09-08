"""``next_symbol`` — local statistical regularity.

Symbols, grounding, and elementary language.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Rec, Tok
from ..lesson import Lesson


def gen_next_symbol(rng: random.Random, ctx):
    """A hidden stochastic bigram grammar; the agent must infer it in-episode."""
    alphabet = list("abcd")
    table = {a: rng.choice(alphabet) for a in alphabet}
    seq = [rng.choice(alphabet)]
    for _ in range(rng.randint(*ctx.span((5, 9), (9, 18)))):
        seq.append(table[seq[-1]])
    obs = Rec(sequence=Lst([Tok(s) for s in seq]), query=Ident("next"))
    return obs, alphabet, table[seq[-1]], {"transition_table": table}


class NextSymbol(Lesson):
    """Local statistical regularity."""

    id = "next_symbol"
    level = 5
    tags = ("symbols", "grounding", "elementary-language")
    teaches = "local statistical regularity"
    capabilities = ()
    axes = {'grammar_complexity': 1}
    answers = ['a', 'b', 'c', 'd']

    generate = staticmethod(gen_next_symbol)
