"""``metalinguistic_reasoning`` — questions about a grammar: grammaticality and ambiguity, by parsing.

Reflective computation and language design.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec, Tok
from ..lesson import Lesson
from ..generators.reflective import _cyk_count, _nonces, _shuffled


def gen_metalinguistic_reasoning(rng: random.Random, ctx):
    """A question *about* a grammar: how many parses does this string have?

    The grammar is invented per episode and is genuinely ambiguous (``S -> S S``
    and friends), so the answer is a count of derivation trees — computed by
    CYK, never annotated. 0 means the string is ungrammatical, 1 unambiguous,
    ≥2 ambiguous, and the target count is drawn uniformly so no constant is
    right more than a quarter of the time.
    """
    target = rng.choice([0, 1, 2, 3])
    fallback = None
    for _ in range(300):
        t0, t1 = _nonces(rng, 2, 2)
        lex = [("A", t0), ("B", t1)]
        for extra in [("A", t1), ("B", t0), ("S", t0), ("S", t1)]:
            if rng.random() < 0.4:
                lex.append(extra)
        pool = [("S", "A", "B"), ("S", "B", "A"), ("S", "S", "S"), ("S", "A", "S"),
                ("S", "S", "B"), ("A", "A", "B"), ("B", "B", "A"), ("S", "B", "B")]
        bins = rng.sample(pool, rng.randint(2, 4))
        if not any(r[0] == "S" for r in bins):
            bins.append(("S", "A", "B"))
        w = [rng.choice([t0, t1]) for _ in range(rng.randint(*ctx.span((3, 5), (6, 8))))]
        n = _cyk_count(w, lex, bins)
        if n > 4:
            continue
        cand = (lex, bins, w, n)
        if fallback is None:
            fallback = cand
        if n == target:
            fallback = cand
            break
    lex, bins, w, n = fallback
    rules = ([Pred("rule", Ident(a), Ident(b), Ident(c)) for a, b, c in bins]
             + [Pred("lex", Ident(a), Tok(t)) for a, t in lex])
    obs = Rec(grammar=Lst(_shuffled(rng, rules)), start=Ident("S"),
              string=Lst([Tok(c) for c in w]), query=Ident("parse_count"))
    return (obs, _shuffled(rng, [0, 1, 2, 3, 4]), n,
            {"parses": n, "string": "".join(w), "n_rules": len(rules)})


class MetalinguisticReasoning(Lesson):
    """Questions about a grammar: grammaticality and ambiguity, by parsing."""

    id = "metalinguistic_reasoning"
    level = 116
    tags = ("reflective-computation", "language-design")
    teaches = "questions about a grammar: grammaticality and ambiguity, by parsing"
    capabilities = ('metareasoning', 'recursive_syntax', 'abstraction')
    axes = {'grammar_complexity': 5, 'recursion_depth': 4, 'ambiguity': 5, 'reasoning_depth': 4}
    answers = [0, 1, 2, 3, 4]

    generate = staticmethod(gen_metalinguistic_reasoning)
