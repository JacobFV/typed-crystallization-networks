"""``representation_invention`` — which new predicate compresses the corpus most.

Ontology and representation.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.ontology import ENTITIES, NEW_PRED_NAMES, REL_NAMES, _description_length, _pattern_instances, _shuffled


def gen_representation_invention(rng: random.Random, ctx):
    """Which new predicate compresses the corpus most?

    A predicate ``P(X,Y) := r1(X,Y) & r2(X,Y)`` costs nine symbols to define and
    saves three for every pair of facts it folds up, so it pays only if it
    recurs. Description length is counted exactly, before and after rewriting;
    the winner is required to be the unique minimum and to beat writing the
    corpus out flat.
    """
    for _ in range(300):
        ents = rng.sample(list(ENTITIES), 6)
        rels = rng.sample(list(REL_NAMES), 4)
        pats = []
        for _ in range(40):
            r1, r2 = rng.sample(rels, 2)
            p = (r1, r2, rng.choice(["same", "rev"]))
            if p not in pats:
                pats.append(p)
            if len(pats) == 4:
                break
        if len(pats) != 4:
            continue
        pairs = [(a, b) for a in ents for b in ents if a != b]
        facts: set = set()
        counts = [rng.randint(4, 6)] + [rng.randint(0, 2) for _ in range(3)]
        for pat, cnt in zip(pats, counts):
            for (x, y) in rng.sample(pairs, cnt):
                r1, r2, orient = pat
                facts.add((r1, x, y))
                facts.add((r2, x, y) if orient == "same" else (r2, y, x))
        for _ in range(rng.randint(*ctx.span((3, 6), (12, 24)))):   # noise to survive
            facts.add((rng.choice(rels), *rng.choice(pairs)))
        # equalize how often each relation occurs: otherwise "name the predicate
        # built from the commonest relations" scores far above chance without
        # anything being matched or measured
        counts = {r: sum(1 for (rr, _x, _y) in facts if rr == r) for r in rels}
        want = max(counts.values())
        for r in rels:
            for _ in range(80):
                if counts[r] >= want:
                    break
                f = (r, *rng.choice(pairs))
                if f in facts:
                    continue
                facts.add(f)
                counts[r] += 1
        if any(counts[r] != want for r in rels):
            continue
        facts = frozenset(facts)
        base = 3 * len(facts)
        dls = [_description_length(facts, p) for p in pats]
        best = min(dls)
        if dls.count(best) != 1 or best >= base:
            continue
        order = sorted(range(4), key=lambda i: dls[i])
        correct = order[0]
        break
    else:                                            # pragma: no cover - construction
        raise RuntimeError("representation_invention: no episode")

    names = _shuffled(rng, NEW_PRED_NAMES)
    answer = names[correct]
    rows = []
    for nm, pat in zip(names, pats):
        r1, r2, orient = pat
        body = [Pred(r1, Ident("X"), Ident("Y")),
                Pred(r2, Ident("X"), Ident("Y")) if orient == "same"
                else Pred(r2, Ident("Y"), Ident("X"))]
        rows.append(Pred("define", Ident(nm), Pred("head", Ident("X"), Ident("Y")), Lst(body)))
    obs = Rec(
        corpus=Lst(_shuffled(rng, [Pred(r, Ident(x), Ident(y)) for (r, x, y) in sorted(facts)])),
        cost_model=Lst([Pred("cost", Ident("atom"), Num(1), Pred("plus_one_per_argument")),
                        Pred("baseline", Num(3 * len(facts)))]),
        candidates=Lst(_shuffled(rng, rows)),
        query=Ident("most_compressive_predicate"),
    )
    hidden = {"baseline": 3 * len(facts), "dl": dict(zip(names, dls)),
              "instances": [len(_pattern_instances(facts, p)) for p in pats], "answer": answer}
    return obs, _shuffled(rng, names), answer, hidden


class RepresentationInvention(Lesson):
    """Which new predicate compresses the corpus most."""

    id = "representation_invention"
    level = 65
    tags = ("ontology", "representation")
    teaches = "which new predicate compresses the corpus most"
    capabilities = ('predicate_invention', 'compression', 'pattern_discovery')
    axes = {'lexical_novelty': 4, 'reasoning_depth': 4, 'compositional_depth': 4, 'world_complexity': 3}

    generate = staticmethod(gen_representation_invention)
