"""``self_error_diagnosis`` — attribute a failure to the stage that caused it.

Self-modeling and architecture adaptation.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.selfmodel import STAGES, _rules, _shuffled


def gen_self_error_diagnosis(rng: random.Random, ctx):
    """Attribute a failure to the stage that caused it.

    The pipeline's stages each declare a deterministic transformation and the
    trace records what each stage actually emitted. Exactly one stage deviates
    from its own specification; every later stage then processes the *corrupted*
    value correctly, so the whole tail of the trace is wrong and only the first
    deviation identifies the culprit. Attribution therefore requires replaying
    the pipeline, not spotting an odd number.
    """
    ops: list[tuple[str, str, int]] = []
    for st in STAGES:
        kind = rng.choice(["add", "sub", "mul"])
        k = 2 if kind == "mul" else rng.randint(*ctx.span((2, 6), (2, 40)))
        ops.append((st, kind, k))
    x = rng.randint(*ctx.span((1, 9), (1, 99)))
    faulty = rng.randrange(len(STAGES))

    outs: list[int] = []
    cur = x
    for i, (_st, kind, k) in enumerate(ops):
        cur = {"add": cur + k, "sub": cur - k, "mul": cur * k}[kind]
        if i == faulty:
            cur += rng.choice([-3, -2, -1, 1, 2, 3])
        outs.append(cur)

    spec = [Pred("stage", Num(i), Ident(st), Ident(kind), Num(k))
            for i, (st, kind, k) in enumerate(ops)]
    trace = [Pred("emitted", Num(i), Ident(ops[i][0]), Num(v)) for i, v in enumerate(outs)]
    obs = Rec(pipeline=Lst(_shuffled(rng, spec)),
              trace=Lst(_shuffled(rng, trace)),
              input=Num(x),
              rules=_rules("stage_0_reads_input_and_stage_i_reads_the_value_stage_i_minus_1_emitted",
                           "a_stage_is_at_fault_iff_its_emitted_value_differs_from_its_op_applied_to_its_input"),
              query=Ident("faulty_stage"))
    return (obs, _shuffled(rng, list(STAGES)), STAGES[faulty],
            {"faulty_index": faulty, "input": x, "outputs": outs})


class SelfErrorDiagnosis(Lesson):
    """Attribute a failure to the stage that caused it."""

    id = "self_error_diagnosis"
    level = 137
    tags = ("self-modeling", "architecture")
    teaches = "attribute a failure to the stage that caused it"
    capabilities = ('self_modeling', 'causal_reasoning', 'metareasoning')
    axes = {'reasoning_depth': 4, 'discourse_horizon': 3, 'compositional_depth': 3}
    answers = ['perception', 'memory', 'representation', 'inference', 'planning', 'execution']

    generate = staticmethod(gen_self_error_diagnosis)
