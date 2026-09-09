"""``next_symbol`` — local statistical regularity.

Symbols, grounding, and elementary language.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Rec, Tok
from ..lesson import Lesson


def _cycle(table, start) -> int:
    """The length of the cycle the chain from ``start`` is already inside."""
    seen, x = {}, start
    while x not in seen:
        seen[x] = len(seen)
        x = table[x]
    return len(seen) - seen[x]


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
    # Four symbols give at most 4**4 * 4 = 1024 distinct worlds, so a lookup
    # over training prompts scores 0.765 by memorising the lesson outright.
    alphabet = list("abcdef")
    steps = rng.randint(*ctx.span((5, 9), (9, 18)))
    # An unconstrained table sends most chains into a fixed point within two
    # steps, so the answer equals the symbol at nearly every position and a
    # fixed-offset copy scores 0.863.  Three conditions keep the episode a
    # question: the last symbol is not its own successor, the chain visits at
    # least three symbols, and the transition being asked about is *shown* --
    # without which the answer is not determined by the observation at all.
    # The answer is the symbol ``c`` places back, where ``c`` is the length of
    # the cycle the chain is in.  Leaving ``c`` at whatever a random table gives
    # makes it 2 almost always, and a fixed-offset copy answers 0.61.  Drawing
    # it uniformly -- from the lengths the sequence is actually long enough to
    # display twice -- leaves no offset better than one in four.
    # 2 is excluded because every even offset then also lands on the answer:
    # ``seq[-4]`` is right whenever the cycle is 2 *or* 4, which is why a fixed
    # offset still scored 0.50 with the cycle merely varying.
    want = rng.choice([c for c in (3, 4, 5) if c < steps] or [3])
    table, seq = {}, []
    for _ in range(600):
        table = {a: rng.choice(alphabet) for a in alphabet}
        seq = [rng.choice(alphabet)]
        for _ in range(steps):
            seq.append(table[seq[-1]])
        if (_cycle(table, seq[-1]) == want and len(set(seq)) >= 3
                and seq.count(seq[-1]) >= 2):
            break
    obs = Rec(sequence=Lst([Tok(s) for s in seq]), query=Ident("next"))
    return (obs, alphabet, table[seq[-1]],
            {"transition_table": table, "cycle": _cycle(table, seq[-1])})


class NextSymbol(Lesson):
    """Local statistical regularity."""

    id = "next_symbol"
    level = 5
    tags = ("symbols", "grounding", "elementary-language")
    teaches = "local statistical regularity"
    capabilities = ()
    axes = {'grammar_complexity': 1}
    #: the default draw uses six symbols; ``hardening="none"`` uses the first four
    answers = ['a', 'b', 'c', 'd', 'e', 'f']

    generate = staticmethod(gen_next_symbol)
