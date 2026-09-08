"""``problem_formulation`` — recovering objective and constraints from outcomes.

Problem formulation and hierarchical agency.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.epistemics import ATTRS, _labelled, _nonces, _shuffled, _spec_accepts


def gen_problem_formulation(rng: random.Random, ctx):
    """The scenario never states its objective; the outcomes do.

    All that is given is a set of items with attributes and a handful of past
    plans marked acceptable or not. Recovering the problem means recovering the
    constraint pair that generated those labels, and each candidate is executed
    against every labelled plan — the three wrong formalizations are kept only
    if they actually misclassify something, so agreement with the data is the
    only thing that separates them.
    """
    n_items = ctx.at(5, 8, default=5)            # items a plan may draw on
    plan_max = ctx.at(4, 6, default=4)           # items in the largest plan
    for _ in range(400):
        names = _nonces(rng, n_items, 4)
        items = {n: {a: rng.randint(1, 9) for a in ATTRS} for n in names}
        a1, a2 = rng.sample(ATTRS, 2)
        truth = [(rng.choice(["at_least", "at_most"]), a1, rng.randint(6, 18)),
                 (rng.choice(["at_least", "at_most"]), a2, rng.randint(6, 18))]
        plans: list[list[str]] = []
        for _ in range(40):
            p = sorted(rng.sample(names, rng.randint(1, plan_max)))
            if p not in plans:
                plans.append(p)
            if len(plans) == 6:
                break
        labels = [_spec_accepts(items, p, truth) for p in plans]
        if labels.count(True) < 2 or labels.count(False) < 2:
            continue

        mutants: list[list[tuple[str, str, int]]] = []
        for _ in range(60):
            m = [list(t) for t in truth]
            j = rng.randrange(2)
            kind = rng.choice(["sense", "attr", "bound"])
            if kind == "sense":
                m[j][0] = "at_most" if m[j][0] == "at_least" else "at_least"
            elif kind == "attr":
                m[j][1] = [a for a in ATTRS if a not in (truth[0][1], truth[1][1])][0]
            else:
                m[j][2] = m[j][2] + rng.choice([-4, -3, 3, 4])
            mm = [tuple(t) for t in m]
            if mm in mutants or mm == list(truth):
                continue
            if [_spec_accepts(items, p, mm) for p in plans] != labels:
                mutants.append(mm)
            if len(mutants) == 3:
                break
        if len(mutants) == 3:
            break
    else:                                     # pragma: no cover - construction
        pass

    cands = [list(truth)] + [list(m) for m in mutants]
    labs, answer = _labelled(rng, cands, 0)
    entries = [Pred("formalization", Ident(lab),
                    Lst([Pred("require", Ident(s), Ident(a), Num(k)) for s, a, k in c]))
               for lab, c in zip(labs, cands)]
    obs = Rec(items=Lst([Pred("item", Ident(n), *[Pred(a, Num(items[n][a])) for a in ATTRS])
                         for n in names]),
              observed_outcomes=Lst(_shuffled(rng, [
                  Pred("outcome", Lst([Ident(x) for x in p]),
                       Ident("acceptable" if lab else "unacceptable"))
                  for p, lab in zip(plans, labels)])),
              candidates=Lst(_shuffled(rng, entries)),
              query=Ident("which_formalization_fits"))
    return (obs, _shuffled(rng, labs), answer,
            {"truth": [list(t) for t in truth], "answer": answer,
             "n_acceptable": labels.count(True)})


class ProblemFormulation(Lesson):
    """Recovering objective and constraints from outcomes."""

    id = "problem_formulation"
    level = 103
    tags = ("problem-formulation", "hierarchical-agency")
    teaches = "recovering objective and constraints from outcomes"
    capabilities = ('problem_formulation', 'induction', 'constraint_reasoning')
    axes = {'reasoning_depth': 4, 'world_complexity': 4, 'representation_freedom': 3}

    generate = staticmethod(gen_problem_formulation)
