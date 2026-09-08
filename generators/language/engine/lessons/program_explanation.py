"""``program_explanation`` — the inverse direction: program to description.

Analogy, causality, planning, and programs.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.causal import (_dsl_desc, _dsl_lists, _dsl_mutate, _dsl_program,
                                _dsl_run, _dsl_symbol, _dsl_text, _labels, _options)


def gen_program_explanation(rng: random.Random, ctx):
    """The inverse task: given the program, which description says what it does?
    Distractors describe mutants differing by one operator or one bound, and are
    kept only if they compute something different on some probe input — a
    description that is merely worded differently would be a second correct
    answer."""
    n_wrong = ctx.at(3, 6, default=3)          # rival descriptions to rule out
    for _ in range(200):
        prog = _dsl_program(rng)
        probes = _dsl_lists(rng, 6)
        truth = [_dsl_run(prog, xs) for xs in probes]
        if all(len(o) == 0 for o in truth):
            continue
        wrong: list[list[tuple[str, int]]] = []
        texts = {_dsl_text(prog)}
        for _ in range(300):
            m = _dsl_mutate(rng, prog)
            if m == list(prog) or m in wrong or _dsl_text(m) in texts:
                continue
            if any(_dsl_run(m, xs) != out for xs, out in zip(probes, truth)):
                wrong.append(m)
                texts.add(_dsl_text(m))
            if len(wrong) == n_wrong:
                break
        if len(wrong) < n_wrong:
            continue

        opts, correct = _options(rng, list(prog), wrong)
        labels = _labels("d", len(opts))
        obs = Rec(program=Lst(_dsl_symbol(prog)),
                  descriptions=Lst([Pred("description", Ident(lab), _dsl_desc(p))
                                    for lab, p in zip(labels, opts)]),
                  query=Ident("which_description_matches_the_program"))
        hidden = {"program": [list(op) for op in prog], "description": _dsl_text(prog),
                  "answer_label": labels[correct]}
        return obs, labels, labels[correct], hidden

    raise RuntimeError("program_explanation: no admissible episode")


class ProgramExplanation(Lesson):
    """The inverse direction: program to description."""

    id = "program_explanation"
    level = 47
    tags = ("analogy", "causality", "planning", "programs")
    teaches = "the inverse direction: program to description"
    capabilities = ('program_synthesis', 'abstraction', 'metareasoning')
    axes = {'reasoning_depth': 3, 'compositional_depth': 3, 'grammar_complexity': 3, 'ambiguity': 2}
    answers = ['d0', 'd1', 'd2', 'd3']

    generate = staticmethod(gen_program_explanation)
