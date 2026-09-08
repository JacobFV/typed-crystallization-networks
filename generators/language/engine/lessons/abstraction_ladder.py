"""``abstraction_ladder`` — the most specific schema covering every example.

Ontology and representation.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.ontology import ENTITY_ISA, PERSONS, THINGS, VERB_ISA, _generalize_slot, _label_options, _more_specific, _schema_covers, _shuffled, _slot_members


def gen_abstraction_ladder(rng: random.Random, ctx):
    """Of four nested schemas, which is the most specific covering all examples?

    The candidates form a strict chain (each generalizes exactly one slot by one
    step up an ``isa`` link, ending at the wildcard), so "most specific covering
    all" is well defined. Examples are sampled to match the intended rung and to
    break the rung below it, and both facts are re-verified by :func:`_schema_covers`.
    """
    verbs = [v for v in sorted(VERB_ISA) if v not in set(VERB_ISA.values())]
    for _ in range(200):
        ev0 = (rng.choice(verbs), rng.choice(list(PERSONS)),
               rng.choice([p for p in PERSONS]), rng.choice(list(THINGS)))
        if ev0[1] == ev0[2]:
            continue
        chain = [ev0]
        cur = list(ev0)
        ok = True
        for _ in range(3):
            slots = [i for i in range(4) if cur[i] != "_"]
            if not slots:
                ok = False
                break
            i = rng.choice(slots)
            isa = VERB_ISA if i == 0 else ENTITY_ISA
            cur[i] = _generalize_slot(isa, cur[i])
            chain.append(tuple(cur))
        if not ok or len(set(chain)) != 4:
            continue
        if not all(_more_specific(chain[i], chain[i + 1]) for i in range(3)):
            continue
        j = rng.randrange(4)
        doms = (verbs, PERSONS, PERSONS, THINGS)
        isas = (VERB_ISA, ENTITY_ISA, ENTITY_ISA, ENTITY_ISA)
        pools = [_slot_members(isas[i], chain[j][i], doms[i]) for i in range(4)]
        if any(not p for p in pools):
            continue
        examples = None
        for _ in range(200):
            evs = [tuple(rng.choice(pools[i]) for i in range(4))
                   for _ in range(ctx.at(3, 8, default=3))]
            if any(e[1] == e[2] for e in evs):
                continue
            if not all(_schema_covers(chain[j], e) for e in evs):
                continue
            if j > 0 and all(_schema_covers(chain[j - 1], e) for e in evs):
                continue
            examples = evs
            break
        if examples is None:
            continue
        covering = [i for i in range(4) if all(_schema_covers(chain[i], e) for e in examples)]
        if not covering or min(covering) != j:
            continue
        break
    else:                                            # pragma: no cover - construction
        raise RuntimeError("abstraction_ladder: no episode")

    pairs, vocab, answer = _label_options(rng, chain[j], [c for i, c in enumerate(chain) if i != j])
    isa_facts = [Pred("isa", Ident(a), Ident(b)) for a, b in sorted(VERB_ISA.items())]
    isa_facts += [Pred("isa", Ident(a), Ident(b)) for a, b in sorted(ENTITY_ISA.items())]
    obs = Rec(
        taxonomy=Lst(_shuffled(rng, isa_facts)),
        examples=Lst([Pred("event", *[Ident(s) for s in e]) for e in examples]),
        schemas=Lst([Pred("schema", Ident(lab), Pred("event", *[Ident(s) for s in c]))
                     for lab, c in pairs]),
        query=Ident("most_specific_covering_schema"),
    )
    hidden = {"rung": j, "chain": [list(c) for c in chain], "answer": answer}
    return obs, vocab, answer, hidden


class AbstractionLadder(Lesson):
    """The most specific schema covering every example."""

    id = "abstraction_ladder"
    level = 66
    tags = ("ontology", "representation")
    teaches = "the most specific schema covering every example"
    capabilities = ('abstraction', 'subsumption', 'generalization_control')
    axes = {'compositional_depth': 4, 'reasoning_depth': 3, 'recursion_depth': 2, 'world_complexity': 2}

    generate = staticmethod(gen_abstraction_ladder)
