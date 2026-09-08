"""``knowledge_gap_detection`` — known / inferable / unknown / malformed.

Epistemics, argument, and teaching.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.epistemics import GAP_LABELS, _nonces, _shuffled


def gen_knowledge_gap_detection(rng: random.Random, ctx):
    """Before answering, say what kind of question this is.

    Four statuses, all of which occur: the fact is stated outright; it follows
    from the stated implications; it is well-formed but nothing in the knowledge
    base settles it; or it is not well-formed at all because it mentions an
    entity or property outside the declared ontology. Collapsing "unknown" into
    "no" and "malformed" into "unknown" are the two failures this measures.
    """
    label = rng.choice(GAP_LABELS)
    for _ in range(400):
        ents = _nonces(rng, 4, 4)
        props = _nonces(rng, ctx.at(7, 14, default=7), 4, avoid=ents)
        implies = []
        for i, p in enumerate(props):
            if i and rng.random() < 0.55:
                implies.append((rng.choice(props[:i]), p))
        base = {e: sorted({p for p in props if rng.random() < 0.25}) for e in ents}

        def closure(e: str) -> set[str]:
            have = set(base[e])
            changed = True
            while changed:
                changed = False
                for a, b in implies:
                    if a in have and b not in have:
                        have.add(b)
                        changed = True
            return have

        e = rng.choice(ents)
        cl = closure(e)
        if label == "known":
            pool = sorted(base[e])
        elif label == "inferable":
            pool = sorted(cl - set(base[e]))
        elif label == "unknown":
            pool = sorted(set(props) - cl)
        else:
            pool = ["*"]
        if not pool:
            continue
        if label == "malformed":
            bad = _nonces(rng, 1, 5, avoid=ents + props)[0]
            if rng.random() < 0.5:
                qe, qp = bad, rng.choice(props)
            else:
                qe, qp = e, bad
        else:
            qe, qp = e, rng.choice(pool)
        break
    else:                                     # pragma: no cover - construction
        qe, qp = e, rng.choice(props)

    obs = Rec(ontology=Lst(_shuffled(rng, [Pred("entity", Ident(x)) for x in ents]
                                     + [Pred("property", Ident(x)) for x in props])),
              facts=Lst(_shuffled(rng, [Pred("has", Ident(x), Ident(p))
                                        for x in ents for p in base[x]])),
              implications=Lst(_shuffled(rng, [Pred("implies", Ident(a), Ident(b))
                                               for a, b in implies])),
              query=Pred("status_of", Pred("has", Ident(qe), Ident(qp))))
    return (obs, _shuffled(rng, GAP_LABELS), label,
            {"label": label, "entity": qe, "property": qp, "n_implications": len(implies)})


class KnowledgeGapDetection(Lesson):
    """Known / inferable / unknown / malformed."""

    id = "knowledge_gap_detection"
    level = 102
    tags = ("epistemics", "argument", "teaching")
    teaches = "known / inferable / unknown / malformed"
    capabilities = ('calibration', 'metacognition', 'ontology_awareness')
    axes = {'reasoning_depth': 4, 'ambiguity': 3, 'uncertainty': 3}
    answers = ['known', 'inferable', 'unknown', 'malformed']

    generate = staticmethod(gen_knowledge_gap_detection)
