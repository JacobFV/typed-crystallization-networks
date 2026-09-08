"""``noisy_channel_language`` — error correction from redundancy.

Analogy, causality, planning, and programs.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec, Str, Tok
from ..lesson import Lesson
from ..generators.social import CONCEPTS, _lev, _shuffled, _spread_code


def gen_noisy_channel_language(rng: random.Random, ctx):
    """A codebook with slack, a message damaged in transit, one recoverable reading.

    The codebook is drawn with pairwise Hamming distance at least three, so a
    single substitution, deletion or transposition leaves the sent word strictly
    nearest to what arrived. That is *checked*, not assumed: the generator scores
    the corrupted string against every codeword and rejects any episode where the
    minimum is tied or lands on the wrong word, so a decoder that reasons about
    the redundancy is exactly right and a decoder that pattern-matches is not.
    """
    alphabet = "abc"
    # the codebook is what makes decoding hard: how many words are in it (capped
    # by the concept list) and how long each codeword is
    n_words = ctx.at(5, 6, default=5)
    length = ctx.at(5, 9, default=5)
    for _ in range(200):
        words = rng.sample(CONCEPTS, n_words)
        code = _spread_code(rng, n_words, length, alphabet, 3)
        if code is None:
            continue
        book = dict(zip(words, code))
        sent = rng.choice(words)
        clean = book[sent]
        kind = rng.choice(["substitute", "delete", "transpose"])
        if kind == "substitute":
            i = rng.randrange(len(clean))
            received = clean[:i] + rng.choice([c for c in alphabet if c != clean[i]]) + clean[i + 1:]
        elif kind == "delete":
            i = rng.randrange(len(clean))
            received = clean[:i] + clean[i + 1:]
        else:
            swaps = [i for i in range(len(clean) - 1) if clean[i] != clean[i + 1]]
            if not swaps:
                continue
            i = rng.choice(swaps)
            received = clean[:i] + clean[i + 1] + clean[i] + clean[i + 2:]
        scores = {w: _lev(received, book[w]) for w in words}
        best = min(scores.values())
        if sorted(scores.values())[1] == best or scores[sent] != best:
            continue                                   # not uniquely recoverable
        obs = Rec(codebook=Lst(_shuffled(rng, [Pred("code", Ident(w), Str(book[w])) for w in words])),
                  received=Lst([Tok(c) for c in received]),
                  query=Ident("intended_word"))
        return (obs, _shuffled(rng, words), sent,
                {"corruption": kind, "clean": clean, "received": received,
                 "distances": dict(scores)})
    raise RuntimeError("noisy_channel_language: no admissible world")


class NoisyChannelLanguage(Lesson):
    """Error correction from redundancy."""

    id = "noisy_channel_language"
    level = 54
    tags = ("analogy", "causality", "planning", "programs")
    teaches = "error correction from redundancy"
    capabilities = ('finite_state_induction', 'abstraction')
    axes = {'grammar_complexity': 3, 'ambiguity': 3, 'reasoning_depth': 3}

    generate = staticmethod(gen_noisy_channel_language)
