"""``explanation`` — audience-relative sufficiency of an explanation.

Epistemics, argument, and teaching.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.epistemics import _concept_graph, _follow, _labelled, _need, _shuffled


def gen_explanation(rng: random.Random, ctx):
    """Which explanation is sufficient *for this listener*.

    Sufficiency is audience-relative and here it is decidable: simulate the
    listener, who already knows a stated set of concepts, walking each candidate
    explanation in order. Distractors are a dropped prerequisite, a step placed
    before the thing it depends on, and a substituted irrelevance — each is
    re-simulated and the episode is rejected unless exactly one candidate
    actually delivers the target.
    """
    n_c = ctx.at(8, 15, default=8)
    for _ in range(400):
        concepts, prereqs = _concept_graph(rng, n_c)
        target = concepts[rng.randrange(4, n_c)]
        known = {c for c in concepts if rng.random() < 0.35 and c != target}
        need = sorted(_need(target, known, prereqs), key=concepts.index)
        if len(need) < 3:
            continue
        # a pair (x, p) inside the explanation with p a prerequisite of x
        pairs = [(x, p) for x in need for p in prereqs[x] if p in need]
        spare = [c for c in concepts if c not in need and c not in known]
        if not pairs or not spare:
            continue
        good = list(need)
        dropped = rng.choice([c for c in good])
        cand_drop = [c for c in good if c != dropped]
        x, p = rng.choice(pairs)
        cand_swap = list(good)
        i, j = cand_swap.index(x), cand_swap.index(p)
        cand_swap[i], cand_swap[j] = cand_swap[j], cand_swap[i]
        swapped_out = rng.choice(good)
        cand_sub = [rng.choice(spare) if c == swapped_out else c for c in good]
        cands = [good, cand_drop, cand_swap, cand_sub]
        ok = [target in _follow(c, known, prereqs)[0] for c in cands]
        if ok == [True, False, False, False] and len({tuple(c) for c in cands}) == 4:
            break
    else:                                     # pragma: no cover - construction
        cands, ok = [good, cand_drop, cand_swap, cand_sub], [True, False, False, False]

    labels, answer = _labelled(rng, cands, 0)
    entries = [Pred("explanation", Ident(lab), Lst([Ident(c) for c in cand]))
               for lab, cand in zip(labels, cands)]
    obs = Rec(prerequisites=Lst(_shuffled(rng, [Pred("requires", Ident(c), Ident(p))
                                                for c in concepts for p in prereqs[c]])),
              listener_knows=Lst([Ident(c) for c in sorted(known)]),
              candidates=Lst(_shuffled(rng, entries)),
              query=Pred("sufficient_explanation_of", Ident(target)))
    return (obs, _shuffled(rng, labels), answer,
            {"target": target, "needed": need, "known": sorted(known), "answer": answer})


class Explanation(Lesson):
    """Audience-relative sufficiency of an explanation."""

    id = "explanation"
    level = 98
    tags = ("epistemics", "argument", "teaching")
    teaches = "audience-relative sufficiency of an explanation"
    capabilities = ('explanation', 'theory_of_mind', 'dependency_reasoning')
    axes = {'reasoning_depth': 4, 'compositional_depth': 3, 'discourse_horizon': 3}

    generate = staticmethod(gen_explanation)
