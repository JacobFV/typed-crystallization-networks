"""``mechanism_discovery`` — causal mechanism behind observationally equal laws.

Scientific induction and model discovery.
"""

from __future__ import annotations

import random
from typing import Callable

from .._structure import Ident, Lst, Num, Pred, Rec, Term
from ..lesson import Lesson
from ..generators.science import _add, _labels, _lin, _mul, _shuffled


def gen_mechanism_discovery(rng: random.Random, ctx):
    """Same predictions, different machinery — until you intervene.

    Every candidate mechanism reproduces the observational table exactly: they
    are *observationally equivalent*, so no amount of passive data separates
    them. They differ in whether ``y`` is produced through the intermediate
    ``m``, around it, or partly both, and the episode reports one experiment in
    which ``m`` is *set* rather than observed. The answer is obtained by running
    each mechanism under that ``do``, which is the whole content of the
    distinction between a predictive law and a causal mechanism.
    """
    extra = ctx.at(0, 2, default=0)                        # further rival mechanisms
    for _ in range(400):
        a1 = rng.choice([-3, -2, -1, 1, 2, 3])
        b1 = rng.randint(-4, 4)
        coeffs = rng.sample([-3, -2, -1, 1, 2, 3], 3 + extra)   # a2 for the two "through m" ones
        a2, a3, a4 = coeffs[:3]
        b2 = rng.randint(-4, 4)
        p, q = a2 * a1, a2 * b1 + b2                        # the shared law y = p·x + q
        c3, d3 = p - a3 * a1, q - a3 * b1
        c4, d4 = p - a4 * a1, q - a4 * b1

        mech: list[tuple[str, Term, Callable[[int, int], int]]] = [
            ("through_m", _add(_mul(Num(a2), Ident("m")), Num(b2)),
             lambda x, m: a2 * m + b2),
            ("direct", _add(_mul(Num(p), Ident("x")), Num(q)),
             lambda x, m: p * x + q),
            ("both_paths", _add(_add(_mul(Num(a3), Ident("m")), _mul(Num(c3), Ident("x"))), Num(d3)),
             lambda x, m: a3 * m + c3 * x + d3),
            ("weak_path", _add(_add(_mul(Num(a4), Ident("m")), _mul(Num(c4), Ident("x"))), Num(d4)),
             lambda x, m: a4 * m + c4 * x + d4),
        ]
        for a5 in coeffs[3:]:                              # same table, another path split
            c5, d5 = p - a5 * a1, q - a5 * b1
            mech.append(("mixed_path",
                         _add(_add(_mul(Num(a5), Ident("m")), _mul(Num(c5), Ident("x"))), Num(d5)),
                         lambda x, m, a=a5, c=c5, d=d5: a * m + c * x + d))
        xs = rng.sample(range(-5, 6), 4)
        rows = [(x, a1 * x + b1, p * x + q) for x in xs]
        if not all(all(f(x, m) == y for x, m, y in rows) for _, _, f in mech):
            continue                                        # observational equivalence
        x0 = rng.choice([v for v in range(-5, 6)])
        v = rng.choice([w for w in range(-8, 9) if w != a1 * x0 + b1])
        outcomes = [f(x0, v) for _, _, f in mech]
        if len(set(outcomes)) != len(mech):
            continue                                        # the experiment must separate all
        true_i = rng.randrange(len(mech))
        order = _shuffled(rng, range(len(mech)))
        labels = _labels("m", len(mech))
        answer = labels[order.index(true_i)]
        obs = Rec(mechanisms=Lst([Pred("mechanism", Ident(labels[j]),
                                       Pred("eq", Ident("m"), _lin(a1, "x", b1)),
                                       Pred("eq", Ident("y"), mech[i][1]))
                                  for j, i in enumerate(order)]),
                  observations=Lst([Pred("observed", Num(x), Num(m), Num(y)) for x, m, y in rows]),
                  columns=Lst([Ident("x"), Ident("m"), Ident("y")]),
                  experiment=Pred("intervention", Pred("set", Ident("x"), Num(x0)),
                                  Pred("do", Ident("m"), Num(v)),
                                  Pred("observed", Ident("y"), Num(outcomes[true_i]))),
                  query=Ident("true_mechanism"))
        hidden = {"kind": mech[true_i][0], "answer": answer, "x0": x0, "m_forced": v,
                  "outcome": outcomes[true_i]}
        return obs, _shuffled(rng, labels), answer, hidden
    raise RuntimeError("mechanism_discovery: no admissible episode")


class MechanismDiscovery(Lesson):
    """Causal mechanism behind observationally equal laws."""

    id = "mechanism_discovery"
    level = 74
    tags = ("science", "induction", "model-discovery")
    teaches = "causal mechanism behind observationally equal laws"
    capabilities = ('causal_reasoning', 'scientific_induction')
    axes = {'reasoning_depth': 5, 'compositional_depth': 3, 'world_complexity': 3, 'ambiguity': 2}

    generate = staticmethod(gen_mechanism_discovery)
