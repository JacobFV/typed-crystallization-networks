"""``few_shot_language_learning`` — a new miniature language per episode.

Language as action.
"""

from __future__ import annotations

import random
from typing import Sequence

from .._structure import Lst, Num, Pred, Rec, Str
from ..lesson import Lesson
from ..generators.base import COLORS
from ..generators.semantics import _nonce_words, _shuffled


def gen_few_shot_language_learning(rng: random.Random, ctx):
    """A whole miniature language, invented in this episode and gone at its end.

    Three nonce primitives denote colours and three nonce function words denote
    string operations (``F x -> x x``, ``x K y -> y x``, ``x B y -> x y x``).
    Support demonstrations show each word once; the query composes them in a way
    the support never did, and asks for one position of the output.
    """
    n_prim = ctx.at(3, 6, default=3)
    for _ in range(200):
        prim_w = _nonce_words(rng, n_prim, 3)
        fn_w = _nonce_words(rng, 3, 4, avoid=prim_w)
        rep, swap, wrap = fn_w
        cols = rng.sample(COLORS, n_prim)
        lex = dict(zip(prim_w, cols))

        def run(expr: Sequence[str]) -> list[str]:
            if len(expr) == 1:
                return [lex[expr[0]]]
            if len(expr) == 2 and expr[0] == rep:
                return [lex[expr[1]], lex[expr[1]]]
            if len(expr) == 3 and expr[1] == swap:
                return [lex[expr[2]], lex[expr[0]]]
            if len(expr) == 3 and expr[1] == wrap:
                return [lex[expr[0]], lex[expr[2]], lex[expr[0]]]
            raise ValueError(expr)                        # pragma: no cover

        support: list[tuple[str, ...]] = [(p,) for p in prim_w]
        support.append((rep, rng.choice(prim_w)))
        a, b = rng.sample(prim_w, 2)
        support.append((a, swap, b))
        a, b = rng.sample(prim_w, 2)
        support.append((a, wrap, b))

        form = rng.choice([rep, swap, wrap])
        if form == rep:
            query_e: tuple[str, ...] = (rep, rng.choice(prim_w))
        else:
            x, y = rng.sample(prim_w, 2)
            query_e = (x, form, y)
        if query_e in support:
            continue
        out = run(query_e)
        k = rng.randrange(len(out))
        break
    else:                                                  # pragma: no cover
        raise RuntimeError("no held-out command")

    demos = _shuffled(rng, [Pred("ex", Str(" ".join(e)), Str(" ".join(run(e))))
                            for e in support])
    obs = Rec(demonstrations=Lst(demos),
              query=Pred("nth_output", Str(" ".join(query_e)), Num(k)))
    return obs, _shuffled(rng, cols), out[k], {"lexicon": lex, "command": " ".join(query_e),
                                               "output": out, "index": k}


class FewShotLanguageLearning(Lesson):
    """A new miniature language per episode."""

    id = "few_shot_language_learning"
    level = 30
    tags = ("pragmatics", "language-as-action")
    teaches = "a new miniature language per episode"
    capabilities = ('ontology_learning', 'abstraction', 'lexical_grounding')
    axes = {'lexical_novelty': 4, 'compositional_depth': 4, 'grammar_complexity': 3}

    generate = staticmethod(gen_few_shot_language_learning)
