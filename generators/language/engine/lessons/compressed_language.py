"""``compressed_language`` — codes, budgets and unique decodability.

Analogy, causality, planning, and programs.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec, Str, Tok
from ..lesson import Lesson
from ..generators.social import TAGS, _code_ok, _encoded_length, _nonce, _prefix_free, _shuffled


def gen_compressed_language(rng: random.Random, ctx):
    """Four candidate codes, one channel budget, exactly one code that works.

    Each rival fails for a different reason and the reasons are the two real
    pressures on a communication protocol: a fixed-width code is unambiguous but
    over budget, and the two short codes fit the budget but are not uniquely
    decodable (one has a codeword that prefixes another, one reuses a codeword for
    two words). Deciding this is exact — check prefix-freeness, then sum lengths
    against the budget — and it is decided by the generator with the same checker
    the learner would have to apply.
    """
    for _ in range(200):
        types = [_nonce(rng, 2) for _ in range(4)]
        if len(set(types)) < 4:
            continue
        message = [rng.choice(types) for _ in range(rng.randint(*ctx.span((6, 9), (16, 22))))]
        if len(set(message)) < 4:
            continue
        freq = {w: message.count(w) for w in types}
        # the longest codeword goes to the most frequent word: deliberately *not*
        # the Huffman assignment, which keeps the budget wide enough that the
        # ambiguous rivals fail on ambiguity alone rather than on length
        ranked = sorted(types, key=lambda w: (-freq[w], w))
        good = dict(zip(ranked, ["111", "110", "10", "0"]))
        budget = _encoded_length(good, message)
        wide = dict(zip(_shuffled(rng, types), ["0000", "0001", "0010", "0011"]))
        prefixy = dict(zip(_shuffled(rng, types), ["0", "01", "1", "11"]))
        collide = dict(zip(_shuffled(rng, types), ["0", "10", "10", "11"]))
        candidates = [good, wide, prefixy, collide]
        if sum(_code_ok(c, message, budget) for c in candidates) != 1 or not _code_ok(good, message, budget):
            continue
        order = _shuffled(rng, list(range(4)))
        tags = rng.sample(TAGS, 4)
        codes = [candidates[i] for i in order]
        answer = tags[order.index(0)]
        obs = Rec(message=Lst([Tok(w) for w in message]),
                  budget=Num(budget),
                  codes=Lst([Pred("code", Ident(tags[k]),
                                  Lst([Pred("bits", Ident(w), Str(codes[k][w])) for w in types]))
                             for k in range(4)]),
                  query=Ident("usable_code"))
        return (obs, _shuffled(rng, tags), answer,
                {"budget": budget, "lengths": [_encoded_length(c, message) for c in codes],
                 "prefix_free": [_prefix_free(list(c.values())) for c in codes],
                 "answer_tag": answer})
    raise RuntimeError("compressed_language: no admissible world")


class CompressedLanguage(Lesson):
    """Codes, budgets and unique decodability."""

    id = "compressed_language"
    level = 53
    tags = ("analogy", "causality", "planning", "programs")
    teaches = "codes, budgets and unique decodability"
    capabilities = ('abstraction', 'program_synthesis')
    axes = {'grammar_complexity': 4, 'reasoning_depth': 4, 'compositional_depth': 3}

    generate = staticmethod(gen_compressed_language)
