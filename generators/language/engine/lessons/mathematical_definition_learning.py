"""``mathematical_definition_learning`` — a predicate defined by axioms, then decided.

Mathematics and formal reasoning.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.mathematics import _cond_sym, _cond_test, _label_items, _nonce, _rand_cond, _shuffled


def gen_mathematical_definition_learning(rng: random.Random, ctx):
    """A nonce predicate is *defined by axioms* in the episode, then applied.

    The definition is a conjunction of two (possibly negated) primitive
    conditions on a finite set of integers. Four candidate sets are drawn and
    kept only if exactly one of them satisfies the definition, so membership is
    a fact about the axioms rather than a judgement call."""
    word = _nonce(rng, 2)
    n_sets = ctx.at(4, 8, default=4)              # candidate sets to test the definition against
    for _ in range(400):
        c1, c2 = _rand_cond(rng), _rand_cond(rng)
        if c1[0] == c2[0]:
            continue
        neg1, neg2 = rng.random() < 0.35, rng.random() < 0.35
        t1, t2 = _cond_test(c1), _cond_test(c2)

        def holds(s, t1=t1, t2=t2, neg1=neg1, neg2=neg2):
            return (t1(s) != neg1) and (t2(s) != neg2)

        sets = [tuple(sorted(rng.sample(range(1, 10), rng.randint(2, 5)))) for _ in range(n_sets)]
        if len(set(sets)) < n_sets:
            continue
        good = [s for s in sets if holds(s)]
        if len(good) != 1:
            continue
        target = good[0]
        others = [s for s in sets if s != target]
        break
    else:                                                # pragma: no cover - construction
        target, others = (2, 4, 6), [(1,), (3,), (5,)]
        c1, c2, neg1, neg2 = ("has_even",), ("size_at_least", 3), False, False

    def lit(cond, neg):
        return Pred("not", _cond_sym(cond)) if neg else _cond_sym(cond)

    shown, label_of = _label_items(rng, [target] + others, prefix="s")
    axiom = Pred("iff", Pred(word, Ident("X")),
                 Pred("and", lit(c1, neg1), lit(c2, neg2)))
    obs = Rec(axioms=Lst([Pred("forall", Ident("X"), axiom)]),
              candidates=Lst([Pred("set", Ident(lab), Lst([Num(x) for x in s]))
                              for lab, s in shown]),
              query=Pred("which_satisfies", Ident(word)))
    answer = label_of[0]
    return obs, _shuffled(rng, [lab for lab, _ in shown]), answer, {
        "predicate": word, "definition": f"{'!' if neg1 else ''}{c1} & {'!' if neg2 else ''}{c2}",
        "member": list(target)}


class MathematicalDefinitionLearning(Lesson):
    """A predicate defined by axioms, then decided."""

    id = "mathematical_definition_learning"
    level = 81
    tags = ("mathematics", "formal-reasoning")
    teaches = "a predicate defined by axioms, then decided"
    capabilities = ('definition_use', 'membership_decision', 'axiom_reading')
    axes = {'lexical_novelty': 4, 'compositional_depth': 3, 'reasoning_depth': 3}

    generate = staticmethod(gen_mathematical_definition_learning)
