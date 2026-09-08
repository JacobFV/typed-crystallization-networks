"""``curriculum_design`` — ordering learning experiences for another learner.

Epistemics, argument, and teaching.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.epistemics import _follow, _labelled, _nonces, _shuffled


def gen_curriculum_design(rng: random.Random, ctx):
    """Ordering is the whole lesson: same material, four sequences.

    A lesson is learned only if its prerequisites are already learned, and a
    lesson presented too early is simply lost. So the four orderings differ in
    how many of the same lessons actually land — computed by simulating each
    sequence against the student's starting knowledge, with a unique maximum
    required.
    """
    for _ in range(400):
        n = rng.randint(*ctx.span((6, 7), (13, 15)))
        lessons = _nonces(rng, n, 4)
        prereqs = {c: sorted(rng.sample(lessons[:i], min(i, rng.choice([0, 1, 1, 2]))))
                   for i, c in enumerate(lessons)}
        known = {c for c in lessons if rng.random() < 0.2}
        orders = []
        for _ in range(40):
            o = _shuffled(rng, lessons)
            if o not in orders:
                orders.append(o)
            if len(orders) == 4:
                break
        if len(orders) < 4:
            continue
        scores = [len(_follow(o, known, prereqs)[0]) - len(known) for o in orders]
        best = max(scores)
        if scores.count(best) == 1 and best - min(scores) >= 2:
            break
    else:                                     # pragma: no cover - construction
        pass

    labels, answer = _labelled(rng, orders, scores.index(best))
    entries = [Pred("ordering", Ident(lab), Lst([Ident(c) for c in o]))
               for lab, o in zip(labels, orders)]
    obs = Rec(prerequisites=Lst(_shuffled(rng, [Pred("requires", Ident(c), Ident(p))
                                                for c in lessons for p in prereqs[c]])),
              student_knows=Lst([Ident(c) for c in sorted(known)]),
              candidates=Lst(_shuffled(rng, entries)),
              query=Ident("ordering_that_teaches_most"))
    return (obs, _shuffled(rng, labels), answer,
            {"scores": dict(zip(labels, scores)), "answer": answer, "n_lessons": len(lessons)})


class CurriculumDesign(Lesson):
    """Ordering learning experiences for another learner."""

    id = "curriculum_design"
    level = 101
    tags = ("epistemics", "argument", "teaching")
    teaches = "ordering learning experiences for another learner"
    capabilities = ('teaching', 'planning', 'dependency_reasoning')
    axes = {'reasoning_depth': 4, 'planning_horizon': 3, 'world_complexity': 3}

    generate = staticmethod(gen_curriculum_design)
