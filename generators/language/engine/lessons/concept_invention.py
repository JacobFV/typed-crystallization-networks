"""``concept_invention`` — inventing the predicate that compresses examples.

Language as action.
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.base import COLORS, SHAPES
from ..generators.semantics import SIZES, _shuffled


def gen_concept_invention(rng: random.Random, ctx):
    """Which invented predicate compresses the examples?

    A hidden two-attribute conjunction generates the positives; the negatives
    each break one conjunct. Of the four candidate predicates exactly one covers
    every positive and no negative — the over-general ones (a single conjunct)
    admit a negative, and the over-specific one (a conjunct on an irrelevant
    attribute) drops a positive. Coverage is checked by evaluation, so the label
    cannot be wrong.
    """
    attrs = {"color": COLORS, "shape": SHAPES, "size": SIZES}
    keys = sorted(attrs)
    for _ in range(200):
        k1, k2, k3 = _shuffled(rng, keys)
        v1, v2 = rng.choice(attrs[k1]), rng.choice(attrs[k2])

        def other(k: str, v: str) -> str:
            return rng.choice([x for x in attrs[k] if x != v])

        pos, neg = [], []
        sizes = rng.sample(attrs[k3], 3)
        for i in range(3):                               # positives vary on k3
            pos.append({k1: v1, k2: v2, k3: sizes[i]})
        neg.append({k1: v1, k2: other(k2, v2), k3: rng.choice(attrs[k3])})
        neg.append({k1: other(k1, v1), k2: v2, k3: rng.choice(attrs[k3])})
        neg.append({k1: other(k1, v1), k2: other(k2, v2), k3: rng.choice(attrs[k3])})

        hyps = [((k1, v1), (k2, v2)),                    # the true concept
                ((k1, v1), None),                        # over-general
                ((k2, v2), None),                        # over-general
                ((k1, v1), (k3, sizes[0]))]              # over-specific
        # further over-specific rivals: each pins the irrelevant attribute to one
        # positive's value, so it still drops the other two positives
        spare = [((k2, v2), (k3, sizes[0])), ((k1, v1), (k3, sizes[1])),
                 ((k2, v2), (k3, sizes[1])), ((k1, v1), (k3, sizes[2])),
                 ((k2, v2), (k3, sizes[2]))]
        hyps += spare[:ctx.at(0, 5, default=0)]

        def covers(h: Sequence[Any], item: Mapping[str, str]) -> bool:
            return all(item[c[0]] == c[1] for c in h if c)

        good = [i for i, h in enumerate(hyps)
                if all(covers(h, p) for p in pos) and not any(covers(h, n) for n in neg)]
        if good == [0]:
            break
    else:                                                  # pragma: no cover
        raise RuntimeError("no separating hypothesis")

    hids = _shuffled(rng, [f"h{i + 1}" for i in range(len(hyps))])
    order = _shuffled(rng, list(range(len(hyps))))
    answer = hids[order.index(0)]
    hyp_syms = []
    for slot, hi in enumerate(order):
        c = hyps[hi]
        a1 = c[0]
        a2 = c[1] if c[1] else ("any", "any")
        hyp_syms.append(Pred("hypothesis", Ident(hids[slot]), Ident(a1[0]), Ident(a1[1]),
                             Ident(a2[0]), Ident(a2[1])))

    items = [Pred("example", Ident(f"p{i}"), Ident(p["color"]), Ident(p["shape"]),
                  Ident(p["size"]), Ident("positive")) for i, p in enumerate(pos)]
    items += [Pred("example", Ident(f"n{i}"), Ident(p["color"]), Ident(p["shape"]),
                   Ident(p["size"]), Ident("negative")) for i, p in enumerate(neg)]
    obs = Rec(examples=Lst(_shuffled(rng, items)),
              hypotheses=Lst(hyp_syms),
              query=Ident("which_predicate_compresses_the_examples"))
    return obs, _shuffled(rng, hids), answer, {"concept": [[k1, v1], [k2, v2]],
                                               "irrelevant_attribute": k3}


class ConceptInvention(Lesson):
    """Inventing the predicate that compresses examples."""

    id = "concept_invention"
    level = 40
    tags = ("pragmatics", "language-as-action")
    teaches = "inventing the predicate that compresses examples"
    capabilities = ('abstraction', 'open_ended_discovery', 'scientific_induction')
    axes = {'reasoning_depth': 4, 'compositional_depth': 4, 'lexical_novelty': 3}

    generate = staticmethod(gen_concept_invention)
