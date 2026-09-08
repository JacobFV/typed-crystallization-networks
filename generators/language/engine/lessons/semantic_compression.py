"""``semantic_compression`` — smallest theory that reproduces the observations.

Self-modeling and architecture adaptation.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec, Str, Term
from ..lesson import Lesson
from ..generators.selfmodel import _labels, _predict, _rules, _shuffled


def gen_semantic_compression(rng: random.Random, ctx):
    """Compress the observations into the smallest theory that still reproduces them.

    The literal table of observations is always among the candidates and always
    correct, so the pressure is entirely toward the shorter generative theory —
    but a shorter candidate that mispredicts even one point is not admissible.
    Sizes are stated, predictions are computed, so "smallest lossless theory" is
    exact.
    """
    for _ in range(60):
        family = rng.choice(["linear", "square"])
        a, b = rng.randint(2, 7), rng.randint(-6, 9)
        xs = rng.sample(range(0, 13), ctx.at(5, 12, default=5))
        data = [(x, _predict(family, a, b, x)) for x in xs]
        alt = "square" if family == "linear" else "linear"
        cands = [
            (family, a, b, 3 if family == "linear" else 2),
            (alt, rng.randint(2, 7), rng.randint(-6, 9), 3 if alt == "linear" else 2),
            (family, a + rng.choice([-2, -1, 1, 2]), b + rng.choice([-2, 2]),
             3 if family == "linear" else 2),
            ("table", 0, 0, 2 * len(data)),
        ]
        fits = [c[0] == "table" or all(_predict(c[0], c[1], c[2], x) == y for x, y in data)
                for c in cands]
        sizes = [c[3] for c in cands]
        good = [i for i in range(4) if fits[i]]
        if good and sum(1 for i in good if sizes[i] == min(sizes[j] for j in good)) == 1:
            break
    best = min(good, key=lambda i: sizes[i])

    ids = _labels(rng, "theory", 4)
    facts: list[Term] = []
    for i, (fam, p1, p2, sz) in enumerate(cands):
        facts.append(Pred("theory", Ident(ids[i]), Ident(fam), Num(p1), Num(p2)))
        facts.append(Pred("theory_size", Ident(ids[i]), Num(sz)))
        if fam == "table":
            facts += [Pred("entry", Ident(ids[i]), Num(x), Num(y)) for x, y in data]
    obs = Rec(data=Lst(_shuffled(rng, [Pred("obs", Num(x), Num(y)) for x, y in data])),
              candidates=Lst(_shuffled(rng, facts)),
              semantics=Lst([Pred("form", Ident("linear"), Str("y = p1 * x + p2")),
                             Pred("form", Ident("square"), Str("y = x * x + p2")),
                             Pred("form", Ident("constant"), Str("y = p2")),
                             # its three siblings are formulas, which need no translating;
                             # this one was an English sentence, so it is a
                             # formula too
                             Pred("form", Ident("table"), Str("y = entry(x)"))]),
              rules=_rules("a_theory_is_lossless_iff_it_predicts_every_observation_exactly",
                           "choose_the_lossless_theory_of_least_size"),
              query=Ident("smallest_lossless_theory"))
    return (obs, _shuffled(rng, ids), ids[best],
            {"family": family, "params": [a, b], "sizes": {ids[i]: sizes[i] for i in range(4)},
             "lossless": [ids[i] for i in good]})


class SemanticCompression(Lesson):
    """Smallest theory that reproduces the observations."""

    id = "semantic_compression"
    level = 144
    tags = ("self-modeling", "architecture")
    teaches = "smallest theory that reproduces the observations"
    capabilities = ('abstraction', 'scientific_induction', 'program_synthesis')
    axes = {'reasoning_depth': 4, 'compositional_depth': 3, 'world_complexity': 2}

    generate = staticmethod(gen_semantic_compression)
