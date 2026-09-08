"""``minimum_description_learning`` — theory length traded against unexplained data.

Self-modeling and architecture adaptation.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec, Term
from ..lesson import Lesson
from ..generators.base import COLORS, SHAPES
from ..generators.selfmodel import _labels, _rules, _shuffled


def gen_minimum_description_learning(rng: random.Random, ctx):
    """The explicit MDL trade: theory length plus the cost of listing its exceptions.

    A rule set that covers everything is long; a short one leaves exceptions that
    must be paid for one by one. Because the exception price varies from episode
    to episode, neither the shortest theory nor the most accurate one wins
    reliably — the total has to be summed.
    """
    for _ in range(80):
        items = [{"id": f"i{i}", "color": rng.choice(COLORS[:3]), "shape": rng.choice(SHAPES[:3])}
                 for i in range(ctx.at(8, 22, default=8))]
        for it in items:
            it["label"] = "yes" if rng.random() < 0.5 else "no"
        penalty = rng.choice([2, 3, 4])
        theories = []
        for _t in range(4):
            k = rng.randint(1, 2)
            rules = []
            for _r in range(k):
                attr = rng.choice(["color", "shape"])
                val = rng.choice(COLORS[:3] if attr == "color" else SHAPES[:3])
                rules.append((attr, val))
            theories.append((rules, 2 * len(rules)))
        totals = []
        for rules, size in theories:
            err = sum(1 for it in items
                      if (any(it[a] == v for a, v in rules)) != (it["label"] == "yes"))
            totals.append(size + penalty * err)
        if sorted(totals)[0] != sorted(totals)[1]:
            break
    best = min(range(4), key=lambda i: totals[i])

    ids = _labels(rng, "theory", 4)
    ifacts = [Pred("item", Ident(it["id"]), Ident(it["color"]), Ident(it["shape"]), Ident(it["label"]))
              for it in items]
    tfacts: list[Term] = []
    for i, (rules, size) in enumerate(theories):
        tfacts.append(Pred("theory_size", Ident(ids[i]), Num(size)))
        tfacts += [Pred("predicts_yes_if", Ident(ids[i]), Ident(a), Ident(v)) for a, v in rules]
    obs = Rec(data=Lst(_shuffled(rng, ifacts)),
              candidates=Lst(_shuffled(rng, tfacts)),
              cost_model=Lst([Pred("exception_cost", Num(penalty))]),
              rules=_rules("a_theory_predicts_yes_for_an_item_iff_some_of_its_conditions_matches_that_item",
                           "an_exception_is_an_item_whose_prediction_differs_from_its_label",
                           "total_cost_is_theory_size_plus_exception_cost_times_number_of_exceptions",
                           "choose_the_theory_of_least_total_cost"),
              query=Ident("mdl_optimal_theory"))
    return (obs, _shuffled(rng, ids), ids[best],
            {"totals": {ids[i]: totals[i] for i in range(4)}, "penalty": penalty})


class MinimumDescriptionLearning(Lesson):
    """Theory length traded against unexplained data."""

    id = "minimum_description_learning"
    level = 145
    tags = ("self-modeling", "architecture")
    teaches = "theory length traded against unexplained data"
    capabilities = ('abstraction', 'scientific_induction', 'metareasoning')
    axes = {'reasoning_depth': 4, 'world_complexity': 3, 'compositional_depth': 3}

    generate = staticmethod(gen_minimum_description_learning)
