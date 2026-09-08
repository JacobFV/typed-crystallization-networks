"""``recursive_self_application`` — a program that transforms programs, applied to itself.

Reflective computation and language design.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec, Tok
from ..lesson import Lesson
from ..generators.reflective import _ARITH, _nonces, _num_options, _rewrite, _run_prog


def gen_recursive_self_application(rng: random.Random, ctx):
    """A program that transforms programs, applied to itself.

    ``R`` is a token rewriter; it is *also* a token sequence (its rules, flat),
    so ``R(R)`` is well defined and yields a different rewriter ``R'``. The
    question asks for the value of ``R'(P)`` run on an input — three steps of
    reflection, each of which is executed rather than described. The wrong
    answers are exactly the values produced by stopping early: ``R(P)``, ``P``
    itself, and ``R(R(P))``.
    """
    fallback = None
    for _ in range(400):
        toks = _nonces(rng, 4)
        sem: dict[str, tuple[str, int]] = {}
        for t in toks:
            kind = rng.choice(_ARITH)
            sem[t] = (kind, rng.choice([2, 3]) if kind == "mul" else rng.randint(1, 6))
        m = rng.randint(2, 3)
        keys = rng.sample(toks, m)
        vals = [rng.choice(toks) for _ in keys]
        rules = list(zip(keys, vals))
        flat = [t for pair in rules for t in pair]
        flat2 = _rewrite(rules, flat)
        rules2 = [(flat2[i], flat2[i + 1]) for i in range(0, len(flat2), 2)]
        if len({a for a, _ in rules2}) != len(rules2):     # R' must be a function
            continue
        prog = [rng.choice(toks) for _ in range(rng.randint(*ctx.span((3, 4), (8, 10))))]
        x = rng.randint(1, 6)
        ans = _run_prog(_rewrite(rules2, prog), sem, x)
        near = [_run_prog(_rewrite(rules, prog), sem, x),                       # stopped at R(P)
                _run_prog(prog, sem, x),                                        # no rewrite at all
                _run_prog(_rewrite(rules, _rewrite(rules, prog)), sem, x)]      # R applied twice
        cand = (toks, sem, rules, flat, rules2, prog, x, ans, near)
        if fallback is None:
            fallback = cand
        if ans not in near and len(set(near)) == 3:
            fallback = cand
            break
    toks, sem, rules, flat, rules2, prog, x, ans, near = fallback
    obs = Rec(
        semantics=Lst([Pred("op", Ident(t), Ident(sem[t][0]), Num(sem[t][1])) for t in toks]),
        transformer=Lst([Pred("rule", Ident(a), Ident(b)) for a, b in rules]),
        encoding=Lst([Tok(t) for t in flat]),        # R written as the program it is
        program=Lst([Tok(t) for t in prog]),
        query=Pred("run", Pred("apply", Pred("apply", Ident("R"), Ident("R")), Ident("P")), Num(x)),
    )
    hidden = {"rules": [list(r) for r in rules], "self_applied": [list(r) for r in rules2],
              "program": list(prog), "input": x, "answer": ans}
    return obs, _num_options(rng, ans, near, 4), ans, hidden


class RecursiveSelfApplication(Lesson):
    """A program that transforms programs, applied to itself."""

    id = "recursive_self_application"
    level = 115
    tags = ("reflective-computation", "language-design")
    teaches = "a program that transforms programs, applied to itself"
    capabilities = ('program_synthesis', 'metareasoning', 'abstraction')
    axes = {'recursion_depth': 5, 'compositional_depth': 5, 'reasoning_depth': 5, 'lexical_novelty': 3}

    generate = staticmethod(gen_recursive_self_application)
