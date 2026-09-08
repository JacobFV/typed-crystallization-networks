"""``teaching`` — the example that most enlarges what a student can derive.

Epistemics, argument, and teaching.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.epistemics import _closure, _nonces, _shuffled


def gen_teaching(rng: random.Random, ctx):
    """Which single example moves the student furthest.

    The student's knowledge state and the rule base are both given; teaching one
    fact is worth exactly the size of the deductive closure it unlocks. The four
    candidate facts are all currently underivable, so "teach something new" is
    not discriminating — the answer is the fact that completes the most rules,
    and the episode is rejected unless one candidate strictly dominates.
    """
    for _ in range(400):
        atoms = _nonces(rng, ctx.at(10, 22, default=10), 4)
        rules: list[tuple[tuple[str, ...], str]] = []
        for i, a in enumerate(atoms):
            if i < 2:
                continue
            for _ in range(rng.choice([0, 1, 1, 2])):
                k = rng.choice([1, 2])
                prem = tuple(sorted(rng.sample(atoms[:i], min(k, i))))
                if (prem, a) not in rules:
                    rules.append((prem, a))
        known = {a for a in atoms if rng.random() < 0.3}
        base = _closure(known, rules)
        cands = [a for a in atoms if a not in base]
        if len(cands) < 4:
            continue
        cands = rng.sample(cands, 4)
        gains = [len(_closure(known | {c}, rules)) - len(base) for c in cands]
        best = max(gains)
        if gains.count(best) == 1 and best >= 2:
            break
    else:                                     # pragma: no cover - construction
        pass

    answer = cands[gains.index(best)]
    obs = Rec(rules=Lst(_shuffled(rng, [Pred("rule", Lst([Ident(p) for p in prem]), Ident(c))
                                        for prem, c in rules])),
              student_knows=Lst([Ident(a) for a in sorted(known)]),
              candidate_examples=Lst([Pred("teach", Ident(c)) for c in _shuffled(rng, cands)]),
              query=Ident("most_informative_example"))
    return (obs, _shuffled(rng, cands), answer,
            {"gains": dict(zip(cands, gains)), "closure_size": len(base), "answer": answer})


class Teaching(Lesson):
    """The example that most enlarges what a student can derive."""

    id = "teaching"
    level = 100
    tags = ("epistemics", "argument", "teaching")
    teaches = "the example that most enlarges what a student can derive"
    capabilities = ('teaching', 'theory_of_mind', 'forward_chaining')
    axes = {'reasoning_depth': 4, 'world_complexity': 3, 'compositional_depth': 3}

    generate = staticmethod(gen_teaching)
