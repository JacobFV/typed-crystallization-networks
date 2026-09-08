"""``program_synthesis`` — specification by example to a program in a small DSL.

Analogy, causality, planning, and programs.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec, Term
from ..lesson import Lesson
from ..generators.causal import _dsl_lists, _dsl_mutate, _dsl_program, _dsl_run, _dsl_symbol, _labels, _options


def gen_program_synthesis(rng: random.Random, ctx):
    """Input/output examples from a hidden DSL program; exactly one candidate
    reproduces *every* example. Each distractor is checked to disagree with the
    truth on at least one example actually shown, so consistency decides."""
    for _ in range(200):
        prog = _dsl_program(rng)
        inputs = _dsl_lists(rng, ctx.at(3, 9, default=3))
        outputs = [_dsl_run(prog, xs) for xs in inputs]
        if any(len(o) == 0 for o in outputs):
            continue
        wrong: list[list[tuple[str, int]]] = []
        for _ in range(300):
            m = _dsl_mutate(rng, prog)
            if m == list(prog) or m in wrong:
                continue
            if any(_dsl_run(m, xs) != out for xs, out in zip(inputs, outputs)):
                wrong.append(m)
            if len(wrong) == 3:
                break
        if len(wrong) < 3:
            continue

        opts, correct = _options(rng, list(prog), wrong)
        labels = _labels("p", len(opts))
        ex: list[Term] = []
        for i, (xs, out) in enumerate(zip(inputs, outputs)):
            for j, x in enumerate(xs):
                ex.append(Pred("input", Num(i), Num(j), Num(x)))
            for j, x in enumerate(out):
                ex.append(Pred("output", Num(i), Num(j), Num(x)))
        cands: list[Term] = []
        for lab, p in zip(labels, opts):
            cands.extend(_dsl_symbol(p, lab))
        obs = Rec(examples=Lst(ex), candidates=Lst(cands),
                  query=Ident("which_program_fits_every_example"))
        hidden = {"program": [list(op) for op in prog], "answer_label": labels[correct]}
        return obs, labels, labels[correct], hidden

    raise RuntimeError("program_synthesis: no admissible episode")


class ProgramSynthesis(Lesson):
    """Specification by example to a program in a small DSL."""

    id = "program_synthesis"
    level = 46
    tags = ("analogy", "causality", "planning", "programs")
    teaches = "specification by example to a program in a small DSL"
    capabilities = ('program_synthesis', 'abstraction', 'proof_search')
    axes = {'reasoning_depth': 4, 'compositional_depth': 3, 'lexical_novelty': 2, 'grammar_complexity': 3}
    answers = ['p0', 'p1', 'p2', 'p3']

    generate = staticmethod(gen_program_synthesis)
