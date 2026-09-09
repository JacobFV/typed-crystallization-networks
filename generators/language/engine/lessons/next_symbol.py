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
    if not ctx.hardens("next_symbol"):
        table = {a: rng.choice(alphabet) for a in alphabet}
        seq = [rng.choice(alphabet)]
        for _ in range(rng.randint(*ctx.span((5, 9), (9, 18)))):
            seq.append(table[seq[-1]])
        obs = Rec(sequence=Lst([Tok(s) for s in seq]), query=Ident("next"))
        return obs, alphabet, table[seq[-1]], {"transition_table": table}
    steps = rng.randint(*ctx.span((5, 9), (9, 18)))
    # An unconstrained table sends most chains into a fixed point within two
    # steps, so the answer equals the symbol at nearly every position and a
    # fixed-offset copy scores 0.863.  Three conditions keep the episode a
    # question: the last symbol is not its own successor, the chain visits at
    # least three symbols, and the transition being asked about is *shown* --
    # without which the answer is not determined by the observation at all.
    table, seq = {}, []
    for _ in range(400):
        table = {a: rng.choice(alphabet) for a in alphabet}
        seq = [rng.choice(alphabet)]
        for _ in range(steps):
            seq.append(table[seq[-1]])
        if (table[seq[-1]] != seq[-1] and len(set(seq)) >= 3
                and seq.count(seq[-1]) >= 2):
            break
    obs = Rec(sequence=Lst([Tok(s) for s in seq]), query=Ident("next"))
    return (obs, alphabet, table[seq[-1]],
            {"transition_table": table, "cycle": len(set(seq))})


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
