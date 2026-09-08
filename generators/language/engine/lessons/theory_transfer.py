"""``theory_transfer`` — apply a source-domain law in an unobserved target domain.

Open-ended epistemology.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.selfmodel import _labels, _rules, _shuffled


def gen_theory_transfer(rng: random.Random, ctx):
    """Carry a law from a mapped domain into one whose edges were never observed.

    The correspondence is given; the second domain's relations are not. The only
    way to answer is to pull the query back through the mapping, apply the law
    learned in the source domain, and push the result forward again — the
    minimal, exactly checkable form of reusing a theory.
    """
    n = ctx.at(4, 8, default=4)
    a_ids = _labels(rng, "a", n)
    b_ids = _labels(rng, "b", n)
    mapping = list(rng.sample(range(n), n))       # source index -> target index
    laws = {}
    for name in ("causes", "inhibits"):
        while True:
            f = list(rng.sample(range(n), n))
            if all(f[i] != i for i in range(n)):
                laws[name] = f
                break
    rel = rng.choice(["causes", "inhibits"])
    j = rng.randrange(n)                          # a target-domain element
    src = mapping.index(j)
    answer = b_ids[mapping[laws[rel][src]]]

    a_facts = [Pred(name, Ident(a_ids[i]), Ident(a_ids[f[i]]))
               for name, f in laws.items() for i in range(n)]
    m_facts = [Pred("corresponds", Ident(a_ids[i]), Ident(b_ids[mapping[i]])) for i in range(n)]
    obs = Rec(source_domain=Lst(_shuffled(rng, a_facts)),
              correspondence=Lst(_shuffled(rng, m_facts)),
              target_domain=Lst([Pred("element", Ident(b)) for b in _shuffled(rng, b_ids)]),
              rules=_rules("the_target_domain_obeys_the_same_laws_as_the_source_domain_under_the_correspondence",
                           "no_target_domain_relation_is_observed_directly"),
              query=Pred("holds_in_target", Ident(rel), Ident(b_ids[j])))
    return (obs, _shuffled(rng, b_ids), answer,
            {"relation": rel, "mapping": mapping, "law": laws[rel]})


class TheoryTransfer(Lesson):
    """Apply a source-domain law in an unobserved target domain."""

    id = "theory_transfer"
    level = 152
    tags = ("open-ended-epistemology",)
    teaches = "apply a source-domain law in an unobserved target domain"
    capabilities = ('abstraction', 'scientific_induction', 'unification')
    axes = {'reasoning_depth': 4, 'compositional_depth': 4, 'world_complexity': 3}

    generate = staticmethod(gen_theory_transfer)
