"""``self_model`` — predict a described agent's success from its record.

Self-modeling and architecture adaptation.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.selfmodel import SKILLS, _rules, _shuffled, _yesno


def gen_self_model(rng: random.Random, ctx):
    """Predict whether the *described* agent solves a task it has not attempted.

    The agent's competence is not stated directly: it is implied by a log of
    single-module attempts, so the profile has to be reconstructed before the
    conjunction over the new task's requirements can be evaluated. Half the
    failures are missing competence and half are budget overruns, so neither
    channel alone predicts the label.
    """
    n_sk = ctx.at(4, 6, default=4)
    skills = rng.sample(SKILLS, n_sk)
    n_true = rng.choice([2, 3])
    have_idx = set(rng.sample(range(n_sk), n_true))
    have = {s: (i in have_idx) for i, s in enumerate(skills)}
    present = [s for s in skills if have[s]]
    absent = [s for s in skills if not have[s]]
    budget = rng.randint(6, 12)

    want = rng.random() < 0.5
    if want:
        a, b = rng.sample(present, 2)
        cost = rng.randint(2, budget)
    elif rng.random() < 0.5:                      # failure by missing module
        a = rng.choice(absent)
        b = rng.choice([s for s in skills if s != a])
        cost = rng.randint(2, budget)
    else:                                          # failure by budget
        a, b = rng.sample(present, 2)
        cost = rng.randint(budget + 1, budget + 5)
    if rng.random() < 0.5:
        a, b = b, a
    truth = have[a] and have[b] and cost <= budget

    log = [Pred("attempt", Ident(f"log{i}"), Ident(s), Num(1),
                Ident("solved" if have[s] else "failed")) for i, s in enumerate(skills)]
    obs = Rec(agent=Lst([Pred("budget", Num(budget))]),
              history=Lst(_shuffled(rng, log)),
              rules=_rules("each_logged_attempt_used_one_module_and_cost_1",
                           "an_attempt_solves_iff_every_required_module_is_present_and_cost_le_budget"),
              query=Pred("will_succeed", Ident(a), Ident(b), Num(cost)))
    answers, answer = _yesno(rng, truth)
    return obs, answers, answer, {"have": have, "budget": budget, "cost": cost,
                                  "required": [a, b], "succeeds": truth}


class SelfModel(Lesson):
    """Predict a described agent's success from its record."""

    id = "self_model"
    level = 135
    tags = ("self-modeling", "architecture")
    teaches = "predict a described agent's success from its record"
    capabilities = ('self_modeling', 'metareasoning')
    axes = {'reasoning_depth': 3, 'world_complexity': 2, 'compositional_depth': 2}
    answers = ['yes', 'no']

    generate = staticmethod(gen_self_model)
