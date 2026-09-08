"""``multiscale_modeling`` — micro dynamics, macro question.

Scientific induction and model discovery.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.science import _apply_perm, _shuffled


def gen_multiscale_modeling(rng: random.Random, ctx):
    """Microscopic transport, macroscopic question.

    The micro rule (a site permutation — transport that moves stuff without
    creating or destroying it) is never named: it has to be read off the shown
    trajectory, which is possible because site contents are distinct. The
    question is then asked at the coarse scale, several steps past the end of
    the trajectory: the mass of the left half. Getting it requires both scales —
    identify the micro dynamics, run them forward, then coarse-grain — and the
    distractors are the coarse variable's values at *other* times, so the
    macro-level answer alone is not guessable from the trajectory.
    """
    k, shown = ctx.at(6, 14, default=6), 5
    for _ in range(200):
        counts = rng.sample(range(1, 20), k)
        perm = _shuffled(rng, range(k))
        if all(perm[i] == i for i in range(k)):
            continue
        traj = [tuple(counts)]
        for _ in range(shown - 1):
            traj.append(_apply_perm(traj[-1], perm))
        ahead = rng.randint(2, 7)
        future = traj[-1]
        for _ in range(ahead):
            future = _apply_perm(future, perm)
        left = lambda s: sum(s[: k // 2])
        answer = left(future)
        seen = []
        s = tuple(counts)
        for _ in range(14):
            s = _apply_perm(s, perm)
            if left(s) != answer and left(s) not in seen:
                seen.append(left(s))
        while len(seen) < 4:
            cand = answer + rng.choice([-3, -2, -1, 1, 2, 3])
            if cand not in seen and cand != answer:
                seen.append(cand)
        distract = _shuffled(rng, seen)[:4]
        obs = Rec(trajectory=Lst([Lst([Num(v) for v in s]) for s in traj]),
                  query=Pred("macro", Pred("left_half_mass"), Num(shown - 1 + ahead)))
        hidden = {"permutation": list(perm), "steps_ahead": ahead, "total": sum(counts),
                  "answer": answer}
        return obs, _shuffled(rng, [answer] + distract), answer, hidden
    raise RuntimeError("multiscale_modeling: no admissible episode")


class MultiscaleModeling(Lesson):
    """Micro dynamics, macro question."""

    id = "multiscale_modeling"
    level = 75
    tags = ("science", "induction", "model-discovery")
    teaches = "micro dynamics, macro question"
    capabilities = ('abstraction', 'scientific_induction')
    axes = {'reasoning_depth': 4, 'world_complexity': 4, 'discourse_horizon': 3}

    generate = staticmethod(gen_multiscale_modeling)
