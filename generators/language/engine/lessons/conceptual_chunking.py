"""``conceptual_chunking`` — the macro that removes the most steps from a plan library.

Ontology and representation.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec, Tok
from ..lesson import Lesson
from ..generators.ontology import MACRO_NAMES, OPS, _macro_saving, _shuffled


def gen_conceptual_chunking(rng: random.Random, ctx):
    """Which macro removes the most steps from a library of plans?

    Naming a fragment costs its length once and saves ``len - 1`` steps at every
    non-overlapping occurrence, so a fragment used twice is usually not worth a
    name. Savings are recomputed from the final plans (planting can create
    accidental occurrences), and the winner must be the unique maximum and
    strictly positive.
    """
    for _ in range(300):
        macros = []
        for _ in range(40):
            m = tuple(rng.choice(list(OPS)) for _ in range(rng.choice([2, 2, 3])))
            if m not in macros and len(set(m)) == len(m):
                macros.append(m)
            if len(macros) == 4:
                break
        if len(macros) != 4:
            continue
        n_plans = rng.randint(*ctx.span((3, 4), (6, 7)))
        plans: list[list[str]] = [[] for _ in range(n_plans)]
        plants = ([rng.randint(*ctx.span((3, 5), (6, 9)))]
                  + [rng.randint(*ctx.span((0, 2), (1, 4))) for _ in range(3)])
        seq: list[tuple[int, tuple]] = []
        for idx, (m, cnt) in enumerate(zip(macros, plants)):
            seq += [(idx, m)] * cnt
        rng.shuffle(seq)
        for idx, m in seq:
            p = plans[rng.randrange(n_plans)]
            p.extend(m)
        for p in plans:                              # filler so plans are not pure macros
            for _ in range(rng.randint(1, 3)):
                p.insert(rng.randrange(len(p) + 1), rng.choice(list(OPS)))
        if any(len(p) < 4 for p in plans):
            continue
        savings = [_macro_saving(plans, m) for m in macros]
        best = max(savings)
        if savings.count(best) != 1 or best <= 0:
            continue
        correct = savings.index(best)
        break
    else:                                            # pragma: no cover - construction
        raise RuntimeError("conceptual_chunking: no episode")

    names = _shuffled(rng, MACRO_NAMES)
    answer = names[correct]
    obs = Rec(
        plans=Lst([Pred("plan", Ident(f"pl{i}"), Lst([Tok(o) for o in p]))
                   for i, p in enumerate(plans)]),
        candidates=Lst(_shuffled(rng, [Pred("macro", Ident(nm), Lst([Tok(o) for o in m]))
                                       for nm, m in zip(names, macros)])),
        cost_model=Lst([Pred("cost", Ident("step"), Num(1)),
                        Pred("cost", Ident("define_macro"), Ident("one_per_body_step"))]),
        query=Ident("best_macro"),
    )
    hidden = {"savings": dict(zip(names, savings)), "total_steps": sum(len(p) for p in plans),
              "answer": answer}
    return obs, _shuffled(rng, names), answer, hidden


class ConceptualChunking(Lesson):
    """The macro that removes the most steps from a plan library."""

    id = "conceptual_chunking"
    level = 67
    tags = ("ontology", "representation")
    teaches = "the macro that removes the most steps from a plan library"
    capabilities = ('macro_learning', 'compression', 'planning')
    axes = {'reasoning_depth': 4, 'discourse_horizon': 3, 'compositional_depth': 3}

    generate = staticmethod(gen_conceptual_chunking)
