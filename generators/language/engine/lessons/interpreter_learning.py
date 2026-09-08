"""``interpreter_learning`` — infer a new language's semantics from demonstrations, then run a held-out program.

Reflective computation and language design.
"""

from __future__ import annotations

import random
from itertools import permutations
from typing import Sequence

from .._structure import Ident, Lst, Num, Pred, Rec, Term
from ..lesson import Lesson
from ..generators.reflective import _SEM_KINDS, _nonces, _num_options, _run_ops, _shuffled


def gen_interpreter_learning(rng: random.Random, ctx):
    """Infer the semantics of a brand-new language from program/output pairs,
    then run a held-out program.

    Three opcodes are invented per episode and each is given one of six possible
    meanings. The demonstrations are *checked for identifiability*: of the 120
    injective meaning assignments, exactly one reproduces every demonstrated
    output, so the episode has a single interpreter and the held-out program has
    a single value. The distractors are the outputs the ruled-out interpreters
    would have produced, so guessing the wrong semantics is punished exactly.
    """
    fallback = None
    for _ in range(300):
        ops = _nonces(rng, 3, 3)
        kinds = rng.sample(_SEM_KINDS, 3)
        true_assign = dict(zip(ops, kinds))
        demos = []
        for _ in range(5):
            p = [(rng.choice(ops), rng.randint(1, 5)) for _ in range(rng.randint(1, 3))]
            xin = rng.randint(1, 9)
            demos.append((p, xin, _run_ops(p, true_assign, xin)))
        consistent = []
        for perm in permutations(_SEM_KINDS, 3):
            a = dict(zip(ops, perm))
            if all(_run_ops(p, a, xi) == o for p, xi, o in demos):
                consistent.append(a)
        query = [(rng.choice(ops), rng.randint(1, 5))
                 for _ in range(rng.randint(*ctx.span((2, 3), (5, 7))))]
        xq = rng.randint(1, 9)
        ans = _run_ops(query, true_assign, xq)
        near: list[int] = []
        for perm in permutations(_SEM_KINDS, 3):
            a = dict(zip(ops, perm))
            v = _run_ops(query, a, xq)
            if v != ans and v not in near and abs(v) < 4000:
                near.append(v)
            if len(near) >= 6:
                break
        cand = (ops, true_assign, demos, query, xq, ans, near)
        if fallback is None:
            fallback = cand
        if len(consistent) == 1 and len(near) >= 3:
            fallback = cand
            break
    ops, true_assign, demos, query, xq, ans, near = fallback

    def _steps(p: Sequence[tuple[str, int]]) -> Term:
        return Lst([Pred("step", Ident(op), Num(a)) for op, a in p])

    obs = Rec(demonstrations=Lst([Pred("demo", _steps(p), Num(xi), Num(o)) for p, xi, o in demos]),
              # the hypothesis space is stated, so the language is *identifiable*
              # from the demonstrations rather than merely constrained by them
              operator_space=Lst([Ident(k) for k in _SEM_KINDS]),
              operator_use=Pred("each_opcode_has", Pred("one_distinct_meaning")),
              program=_steps(query),
              query=Pred("output_for_input", Num(xq)))
    hidden = {"semantics": dict(true_assign), "input": xq, "answer": ans,
              "n_demos": len(demos)}
    return obs, _num_options(rng, ans, _shuffled(rng, near), 4), ans, hidden


class InterpreterLearning(Lesson):
    """Infer a new language's semantics from demonstrations, then run a held-out program."""

    id = "interpreter_learning"
    level = 120
    tags = ("reflective-computation", "language-design")
    teaches = "infer a new language's semantics from demonstrations, then run a held-out program"
    capabilities = ('program_synthesis', 'scientific_induction', 'metareasoning')
    axes = {'lexical_novelty': 5, 'compositional_depth': 5, 'reasoning_depth': 5, 'grammar_complexity': 4}

    generate = staticmethod(gen_interpreter_learning)
