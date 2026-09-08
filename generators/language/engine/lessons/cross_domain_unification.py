"""``cross_domain_unification`` — the element that corresponds across two domains.

Open-ended epistemology.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.selfmodel import _labels, _rigid_pair, _rules, _shuffled


def gen_cross_domain_unification(rng: random.Random, ctx):
    """Two domains, one structure: find the element that corresponds.

    The second domain is literally the image of the first under a permutation the
    generator drew, and the graph is rejected unless its automorphism group is
    trivial — so exactly one correspondence is consistent with the edges and the
    answer is the permutation's value, computable by anyone who finds the
    isomorphism.
    """
    n = ctx.at(5, 7, default=5)
    edges, perm = _rigid_pair(rng, n)
    a_ids = _labels(rng, "a", n)
    b_ids = _labels(rng, "b", n)
    rel_a, rel_b = rng.sample(["flows_to", "drives", "feeds", "pushes"], 2)
    a_facts = [Pred(rel_a, Ident(a_ids[u]), Ident(a_ids[v])) for u, v in edges]
    b_facts = [Pred(rel_b, Ident(b_ids[perm[u]]), Ident(b_ids[perm[v]])) for u, v in edges]
    k = rng.randrange(n)
    obs = Rec(domain_a=Lst(_shuffled(rng, a_facts)),
              domain_b=Lst(_shuffled(rng, b_facts)),
              rules=_rules("the_two_domains_share_one_relational_structure_under_a_bijection_of_their_elements",
                           "the_bijection_is_the_unique_one_that_maps_every_edge_of_a_onto_an_edge_of_b"),
              query=Pred("corresponds_to", Ident(a_ids[k])))
    return (obs, _shuffled(rng, b_ids), b_ids[perm[k]],
            {"edges": [list(e) for e in edges], "relation_a": rel_a, "relation_b": rel_b})


class CrossDomainUnification(Lesson):
    """The element that corresponds across two domains."""

    id = "cross_domain_unification"
    level = 151
    tags = ("open-ended-epistemology",)
    teaches = "the element that corresponds across two domains"
    capabilities = ('abstraction', 'open_ended_discovery', 'unification')
    axes = {'reasoning_depth': 5, 'compositional_depth': 4, 'world_complexity': 3}

    generate = staticmethod(gen_cross_domain_unification)
