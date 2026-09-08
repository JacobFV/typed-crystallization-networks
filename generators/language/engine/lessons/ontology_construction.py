"""``ontology_construction`` — which type hierarchy is consistent with every observation.

Ontology and representation.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.ontology import _assign_entities, _consistent, _inherited, _label_options, _mutations, _onto_facts, _onto_key, _random_hierarchy, _shuffled


def gen_ontology_construction(rng: random.Random, ctx):
    """Which of four type hierarchies explains every observation?

    Observations are an entity's *complete* property set. A hierarchy is
    consistent iff, for every entity, the properties inherited along its type's
    ``sub`` chain are exactly the ones observed — so consistency is set equality
    and the label is computed, not annotated. Distractors are perturbations of
    the true hierarchy that are each verified to be inconsistent.
    """
    n_extra = ctx.at(2, 6, default=2)             # entities beyond one per leaf type
    for _ in range(200):
        own, parent, leaves, props, _ = _random_hierarchy(rng)
        inst = _assign_entities(rng, leaves, n_extra)
        observed = {e: _inherited(own, parent, inst[e]) for e in inst}
        if not _consistent(own, parent, inst, observed):
            continue
        pool, seen = [], {_onto_key(own, parent, inst)}
        for o, pa, ins in _shuffled(rng, _mutations(rng, own, parent, inst)):
            k = _onto_key(o, pa, ins)
            if k in seen or _consistent(o, pa, ins, observed):
                continue
            seen.add(k)
            pool.append((o, pa, ins))
            if len(pool) == 3:
                break
        if len(pool) == 3:
            break
    else:                                            # pragma: no cover - construction
        raise RuntimeError("ontology_construction: no episode")

    pairs, vocab, answer = _label_options(rng, (own, parent, inst), pool)
    cands = [Pred("candidate", Ident(lab), _onto_facts(rng, *c)) for lab, c in pairs]
    obs = Rec(
        entities=Lst([Ident(e) for e in sorted(observed)]),
        observations=Lst(_shuffled(rng, [Pred("observed", Ident(e), Ident(p))
                                         for e in sorted(observed) for p in sorted(observed[e])])),
        candidates=Lst(cands),
        query=Pred("consistent_ontology"),
    )
    hidden = {"n_types": len(own), "n_entities": len(inst),
              "max_props": max(len(_inherited(own, parent, t)) for t in own),
              "answer": answer}
    return obs, vocab, answer, hidden


class OntologyConstruction(Lesson):
    """Which type hierarchy is consistent with every observation."""

    id = "ontology_construction"
    level = 61
    tags = ("ontology", "representation")
    teaches = "which type hierarchy is consistent with every observation"
    capabilities = ('ontology_induction', 'consistency_checking', 'inheritance')
    axes = {'world_complexity': 3, 'reasoning_depth': 4, 'compositional_depth': 3, 'recursion_depth': 2}

    generate = staticmethod(gen_ontology_construction)
