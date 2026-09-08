"""``emergence_discovery`` — which macrostate is predictively closed.

Scientific induction and model discovery.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.science import _closed, _has_repeat, _labels, _macro_eval, _shuffled


def gen_emergence_discovery(rng: random.Random, ctx):
    """Which coarse variable has a life of its own?

    A micro trajectory is shown together with four candidate macrostates. One of
    them is *predictively closed* on that trajectory — every time it takes a
    value it goes to the same next value, so it obeys a law at its own level —
    and the other three are not: each revisits a value and then does something
    different, which is exactly the evidence that they leak information to the
    micro scale. Every candidate is forced to revisit a value at least once, so
    closure is never true vacuously, and the check is run on the *shown*
    trajectory, so the answer is a fact about the observation.
    """
    k = ctx.at(5, 9, default=5)
    for _ in range(400):
        m = rng.choice([3, 4, 5])
        alpha, beta = rng.randrange(1, m), rng.randrange(1, m)
        state = tuple(rng.randrange(m) for _ in range(k))
        traj = [state]
        for _ in range(13):
            s = traj[-1]
            traj.append(tuple((alpha * s[i] + beta * s[(i + 1) % k]) % m for i in range(k)))
        specs = [Pred("sum_mod", Num(m)), Pred("sum_mod", Num(2)), Pred("count_equal", Num(0)),
                 Pred("count_gt", Num(1)), Pred("cell", Num(rng.randrange(k))), Pred("max"),
                 Pred("min"), Pred("range"), Pred("half_sum_mod", Num(3))]
        usable, closed = [], []
        for sp in specs:
            seq = [_macro_eval(sp, s) for s in traj]
            if len(set(seq)) < 2 or not _has_repeat(seq):
                continue                                   # constant or never revisits: no test
            usable.append(sp)
            if _closed(seq):
                closed.append(sp)
        if len(closed) != 1 or len(usable) < 4:
            continue
        good = closed[0]
        rest = _shuffled(rng, [sp for sp in usable if sp is not good])[:3]
        if len(rest) < 3:
            continue
        cands = _shuffled(rng, [good] + rest)
        labels = _labels("v", 4)
        answer = labels[next(j for j, sp in enumerate(cands) if sp is good)]
        obs = Rec(trajectory=Lst([Lst([Num(v) for v in s]) for s in traj]),
                  macrostates=Lst([Pred("macrostate", Ident(labels[j]), sp)
                                   for j, sp in enumerate(cands)]),
                  query=Ident("predictively_closed"))
        hidden = {"modulus": m, "alpha": alpha, "beta": beta, "closed": str(good),
                  "answer": answer}
        return obs, _shuffled(rng, labels), answer, hidden
    raise RuntimeError("emergence_discovery: no admissible episode")


class EmergenceDiscovery(Lesson):
    """Which macrostate is predictively closed."""

    id = "emergence_discovery"
    level = 76
    tags = ("science", "induction", "model-discovery")
    teaches = "which macrostate is predictively closed"
    capabilities = ('abstraction', 'scientific_induction', 'open_ended_discovery')
    axes = {'reasoning_depth': 5, 'world_complexity': 4, 'compositional_depth': 3, 'discourse_horizon': 3}

    generate = staticmethod(gen_emergence_discovery)
