"""``ontology_alignment`` — corresponding concepts across two independent ontologies.

Ontology and representation.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.ontology import A_CONCEPT_NAMES, B_CONCEPT_NAMES, ENTITIES, _extension, _shuffled


def gen_ontology_alignment(rng: random.Random):
    """Which concept of ontology B is the concept of ontology A?

    Two ontologies are built over the same eight entities from *different*
    attribute pairs, so their decompositions genuinely differ, and their names
    are drawn from disjoint pools, so nothing lexical survives the translation.
    Exactly one B-concept has the same extension as the queried A-concept, and
    at least one other B-concept has the same *cardinality* — cardinality is a
    plausible shortcut, so it is deliberately broken.
    """
    ents = rng.sample(list(ENTITIES), 8)
    bits = [(i >> 2 & 1, i >> 1 & 1, i & 1) for i in range(8)]
    for _ in range(200):
        shared, a_only, b_only = _shuffled(rng, [0, 1, 2])
        sv = rng.choice([0, 1])
        target = _extension(ents, bits, ((shared, sv),))
        a_specs = [((shared, sv),), ((a_only, 1),), ((a_only, 0),),
                   ((shared, 1 - sv), (a_only, 1))]
        b_specs = [((shared, sv),), ((b_only, 1),), ((b_only, 0),),
                   ((shared, 1 - sv),)]
        a_ext = [_extension(ents, bits, s) for s in a_specs]
        b_ext = [_extension(ents, bits, s) for s in b_specs]
        if len(set(a_ext)) != 4 or len(set(b_ext)) != 4:
            continue
        if sum(1 for x in b_ext if x == target) != 1:
            continue
        if sum(1 for x in b_ext if len(x) == len(target)) < 2:
            continue                                 # need a same-size decoy
        break
    else:                                            # pragma: no cover - construction
        raise RuntimeError("ontology_alignment: no episode")

    a_names = _shuffled(rng, A_CONCEPT_NAMES)
    b_names = _shuffled(rng, B_CONCEPT_NAMES)
    answer = b_names[b_ext.index(target)]
    a_query = a_names[0]                             # a_specs[0] is the shared concept

    def concepts(names, exts):
        rows = [Pred("concept", Ident(nm), Lst([Ident(e) for e in _shuffled(rng, sorted(ex))]))
                for nm, ex in zip(names, exts)]
        return Lst(_shuffled(rng, rows))

    obs = Rec(
        world=Lst([Ident(e) for e in _shuffled(rng, ents)]),
        ontology_a=concepts(a_names, a_ext),
        ontology_b=concepts(b_names, b_ext),
        query=Pred("aligns_with", Ident(a_query)),
    )
    hidden = {"target_size": len(target), "a_concept": a_query, "answer": answer,
              "same_size_decoys": sum(1 for x in b_ext if len(x) == len(target)) - 1}
    return obs, _shuffled(rng, b_names), answer, hidden


class OntologyAlignment(Lesson):
    """Corresponding concepts across two independent ontologies."""

    id = "ontology_alignment"
    level = 63
    tags = ("ontology", "representation")
    teaches = "corresponding concepts across two independent ontologies"
    capabilities = ('extensional_reasoning', 'translation', 'ontology_induction')
    axes = {'lexical_novelty': 4, 'world_complexity': 3, 'reasoning_depth': 3, 'ambiguity': 2}

    generate = staticmethod(gen_ontology_alignment)
