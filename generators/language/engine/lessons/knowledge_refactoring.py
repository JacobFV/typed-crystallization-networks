"""``knowledge_refactoring`` — reorganize a memory without losing a derivable fact.

Self-modeling and architecture adaptation.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec, Term
from ..lesson import Lesson
from ..generators.selfmodel import KINDS, PROPS, _closure, _labels, _rules, _shuffled


def gen_knowledge_refactoring(rng: random.Random, ctx):
    """Reorganize a symbolic memory without losing a single derivable fact.

    Five candidate knowledge bases are offered with their statement counts. One
    over-generalizes — a rule that is shorter than the facts it replaces but
    licenses a fact that was never true — and it is the shortest candidate in the
    list, so length alone picks the wrong answer. The closure of each candidate
    has to be computed and compared against the facts that must be preserved.
    """
    kinds = rng.sample(KINDS, 2)
    n_e = 2 * ctx.at(3, 8, default=3)
    ents = _labels(rng, "e", n_e)
    kind_of = {e: (kinds[0] if i < n_e // 2 else kinds[1]) for i, e in enumerate(ents)}
    k0 = [e for e in ents if kind_of[e] == kinds[0]]
    k1 = [e for e in ents if kind_of[e] == kinds[1]]
    pA, pB, pC = rng.sample(PROPS, 3)

    base = {(e, pA) for e in k0} | {(e, pB) for e in k1}
    m = rng.choice([1, 2])
    extras = [(e, pC) for e in k0[:m]]
    required = base | set(extras)

    ids = _labels(rng, "kb", 5)
    cands = [
        (sorted(required), []),                                        # verbatim
        (sorted(extras), [(kinds[0], pA), (kinds[1], pB)]),            # correct + minimal
        (sorted(extras) + [(k0[0], pA)], [(kinds[0], pA), (kinds[1], pB)]),   # redundant
        (sorted(extras) + [(e, pB) for e in k1], [(kinds[0], pA)]),    # half refactored
        ([], [(kinds[0], pA), (kinds[1], pB), (kinds[0], pC)]),        # over-general
    ]
    sizes = [len(f) + len(r) for f, r in cands]
    valid = [i for i in range(5) if _closure(cands[i][0], cands[i][1], kind_of) == required]
    best = min(valid, key=lambda i: sizes[i])
    assert sum(1 for i in valid if sizes[i] == sizes[best]) == 1
    assert best == 1

    facts: list[Term] = []
    for i in range(5):
        facts.append(Pred("kb_size", Ident(ids[i]), Num(sizes[i])))
        facts += [Pred("kb_fact", Ident(ids[i]), Ident(e), Ident(p)) for e, p in cands[i][0]]
        facts += [Pred("kb_rule", Ident(ids[i]), Ident(k), Ident(p)) for k, p in cands[i][1]]
    obs = Rec(taxonomy=Lst(_shuffled(rng, [Pred("kind", Ident(e), Ident(kind_of[e])) for e in ents])),
              required=Lst(_shuffled(rng, [Pred("holds", Ident(e), Ident(p)) for e, p in sorted(required)])),
              candidates=Lst(_shuffled(rng, facts)),
              rules=_rules("kb_rule_k_p_derives_holds_e_p_for_every_e_of_kind_k",
                           "a_kb_is_faithful_iff_its_derivable_facts_equal_the_required_facts_exactly",
                           "choose_the_faithful_kb_of_least_size"),
              query=Ident("shortest_faithful_kb"))
    return (obs, _shuffled(rng, ids), ids[best],
            {"sizes": {ids[i]: sizes[i] for i in range(5)},
             "valid": [ids[i] for i in valid], "extras": m})


class KnowledgeRefactoring(Lesson):
    """Reorganize a memory without losing a derivable fact."""

    id = "knowledge_refactoring"
    level = 143
    tags = ("self-modeling", "architecture")
    teaches = "reorganize a memory without losing a derivable fact"
    capabilities = ('ontology_learning', 'abstraction', 'metareasoning')
    axes = {'reasoning_depth': 4, 'compositional_depth': 4, 'world_complexity': 3}

    generate = staticmethod(gen_knowledge_refactoring)
