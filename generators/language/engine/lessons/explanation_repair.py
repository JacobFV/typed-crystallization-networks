"""``explanation_repair`` — naming the missing prerequisite from a failure report.

Epistemics, argument, and teaching.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.epistemics import _concept_graph, _follow, _need, _shuffled


def gen_explanation_repair(rng: random.Random, ctx):
    """The listener reports where the explanation broke; name the missing piece.

    An otherwise correct explanation has one prerequisite removed. The listener
    reports the first step they could not follow — that step has exactly one
    unavailable prerequisite, and repairing the explanation means naming it.
    The distractor vocabulary is drawn from concepts that are equally unknown to
    the listener, so "pick something they do not know" is not enough.
    """
    n_concepts = ctx.at(9, 15, default=9)
    for _ in range(400):
        concepts, prereqs = _concept_graph(rng, n_concepts)
        target = concepts[rng.randrange(4, n_concepts)]
        known = {c for c in concepts if rng.random() < 0.3 and c != target}
        need = sorted(_need(target, known, prereqs), key=concepts.index)
        if len(need) < 3:
            continue
        cands = [(x, p) for x in need for p in prereqs[x] if p in need]
        if not cands:
            continue
        x, missing = rng.choice(cands)
        expl = [c for c in need if c != missing]
        _, stuck = _follow(expl, known, prereqs)
        if not stuck or stuck[0][0] != x or len(stuck[0][1]) != 1 or stuck[0][1][0] != missing:
            continue
        others = [c for c in concepts if c not in known and c != missing and c not in expl]
        if len(others) < 3:
            continue
        vocab = _shuffled(rng, [missing] + rng.sample(others, 3))
        break
    else:                                     # pragma: no cover - construction
        vocab, missing, x, expl = [missing], missing, x, expl

    obs = Rec(prerequisites=Lst(_shuffled(rng, [Pred("requires", Ident(c), Ident(p))
                                                for c in concepts for p in prereqs[c]])),
              listener_knows=Lst([Ident(c) for c in sorted(known)]),
              explanation_given=Lst([Ident(c) for c in expl]),
              listener_report=Pred("did_not_follow", Ident(x)),
              query=Pred("missing_prerequisite_for", Ident(x)))
    return (obs, vocab, missing,
            {"missing": missing, "stuck_at": x, "explanation": expl, "target": target})


class ExplanationRepair(Lesson):
    """Naming the missing prerequisite from a failure report."""

    id = "explanation_repair"
    level = 99
    tags = ("epistemics", "argument", "teaching")
    teaches = "naming the missing prerequisite from a failure report"
    capabilities = ('explanation', 'theory_of_mind', 'diagnosis')
    axes = {'reasoning_depth': 4, 'discourse_horizon': 3, 'ambiguity': 2}

    generate = staticmethod(gen_explanation_repair)
