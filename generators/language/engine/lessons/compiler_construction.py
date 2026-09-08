"""``compiler_construction`` — expand nested macros to primitives, then execute.

Reflective computation and language design.
"""

from __future__ import annotations

import random
from typing import Sequence

from .._structure import Ident, Lst, Num, Pred, Rec, Term
from ..lesson import Lesson
from ..generators.reflective import _expand, _nonces, _num_options, _run_prog


def gen_compiler_construction(rng: random.Random, ctx):
    """Compile a high-level program to primitives, then run it.

    Macros are defined in terms of primitives *and of each other*, and a
    ``times`` construct repeats a token, so the compiled instruction stream is
    several times longer than what is written. Only the final value is graded,
    and the distractors are the values produced by the three natural failures:
    not expanding macros, treating ``times n`` as ``times 1``, and expanding
    only the outermost macro layer.
    """
    max_stream = ctx.at(18, 60, default=18)
    max_mul = ctx.at(4, 12, default=4)
    max_val = ctx.at(3000, 3_000_000, default=3000)
    fallback = None
    for _ in range(400):
        prims = _nonces(rng, 3, 2)
        sem = {prims[0]: ("add", rng.randint(1, 5)), prims[1]: ("sub", rng.randint(1, 5)),
               prims[2]: ("mul", 2)}
        mnames = _nonces(rng, 2, 3)
        while any(m in prims for m in mnames):
            mnames = _nonces(rng, 2, 3)
        body0 = [("tok", rng.choice(prims)) for _ in range(rng.randint(2, 3))]
        if rng.random() < 0.6:
            body0[rng.randrange(len(body0))] = ("rep", rng.randint(2, 3), rng.choice(prims))
        body1: list[tuple] = [("tok", mnames[0])]                    # macro over a macro
        for _ in range(rng.randint(1, 2)):
            body1.append(("tok", rng.choice(prims)))
        rng.shuffle(body1)
        macros = {mnames[0]: body0, mnames[1]: body1}
        prog: list[tuple] = []
        for _ in range(rng.randint(*ctx.span((2, 3), (5, 8)))):
            if rng.random() < 0.7:
                t = rng.choice(mnames)
                prog.append(("rep", rng.randint(2, 2), t) if rng.random() < 0.35 else ("tok", t))
            else:
                prog.append(("tok", rng.choice(prims)))
        stream = _expand(prog, macros)
        if (not stream or len(stream) > max_stream
                or sum(1 for t in stream if sem[t][0] == "mul") > max_mul):
            continue
        x = rng.randint(1, 5)
        ans = _run_prog(stream, sem, x)
        d_nomacro = _run_prog([it[1] if it[0] == "tok" else it[2] for it in prog
                               if (it[1] if it[0] == "tok" else it[2]) in prims], sem, x)
        d_once = _run_prog(_expand([(("tok", it[2]) if it[0] == "rep" else it) for it in prog], macros),
                           sem, x)
        shallow = []
        for it in prog:
            base = it[1] if it[0] == "tok" else it[2]
            n = 1 if it[0] == "tok" else it[1]
            body = macros.get(base)
            step = ([b[1] if b[0] == "tok" else b[2] for b in body] if body else [base])
            shallow += [s for s in step if s in prims] * n
        d_shallow = _run_prog(shallow, sem, x)
        near = [d_once, d_shallow, d_nomacro]
        cand = (prims, sem, mnames, macros, prog, x, ans, near, stream)
        if fallback is None:
            fallback = cand
        if ans not in near and len({ans, *near}) == 4 and abs(ans) < max_val:
            fallback = cand
            break
    prims, sem, mnames, macros, prog, x, ans, near, stream = fallback

    def _items(items: Sequence[tuple]) -> Term:
        return Lst([Ident(it[1]) if it[0] == "tok" else Pred("times", Num(it[1]), Ident(it[2]))
                    for it in items])

    obs = Rec(primitives=Lst([Pred("prim", Ident(p), Ident(sem[p][0]), Num(sem[p][1])) for p in prims]),
              macros=Lst([Pred("macro", Ident(m), _items(macros[m])) for m in mnames]),
              program=_items(prog),
              query=Pred("output_for_input", Num(x)))
    hidden = {"compiled_length": len(stream), "input": x, "answer": ans,
              "compiled": " ".join(stream)}
    return obs, _num_options(rng, ans, near, 4), ans, hidden


class CompilerConstruction(Lesson):
    """Expand nested macros to primitives, then execute."""

    id = "compiler_construction"
    level = 119
    tags = ("reflective-computation", "language-design")
    teaches = "expand nested macros to primitives, then execute"
    capabilities = ('program_synthesis', 'abstraction', 'recursive_syntax')
    axes = {'recursion_depth': 5, 'compositional_depth': 5, 'reasoning_depth': 5}

    generate = staticmethod(gen_compiler_construction)
