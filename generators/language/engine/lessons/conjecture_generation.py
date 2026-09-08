"""``conjecture_generation`` — a claim both true in the structure and not already implied.

Mathematics and formal reasoning.
"""

from __future__ import annotations

import random
from typing import Any

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.mathematics import _cval, _entails, _fsym, _label_items, _rand_formula, _shuffled


def gen_conjecture_generation(rng: random.Random, ctx):
    """A good conjecture is true *and* not already known.

    A concrete structure (a full truth assignment) is generated together with a
    premise set it satisfies. Exactly one of four candidate claims is both true
    in the structure and *underdetermined* by the premises — the others are
    either false in the structure or already entailed, and both conditions are
    checked by enumerating every valuation."""
    atoms = list("abcde")
    depth = ctx.at(2, 4, default=2)
    for _ in range(400):
        model = {a: rng.random() < 0.5 for a in atoms}
        premises = []
        for _ in range(3):
            for _ in range(30):
                f = _rand_formula(rng, atoms, depth)
                if _cval(f, model) and f not in premises:
                    premises.append(f)
                    break
        if len(premises) != 3:
            continue
        pool: list[Any] = []
        for _ in range(60):
            f = _rand_formula(rng, atoms, depth)
            if f not in pool and f not in premises:
                pool.append(f)
        good = [f for f in pool if _cval(f, model) and not _entails(premises, f, atoms)]
        stale = [f for f in pool if _cval(f, model) and _entails(premises, f, atoms)]
        false_ = [f for f in pool if not _cval(f, model)]
        if not good or not stale or len(false_) < 2:
            continue
        answer_f = rng.choice(good)
        cands = [answer_f, rng.choice(stale)] + rng.sample(false_, 2)
        if len(set(cands)) < 4:
            continue
        break
    else:                                                # pragma: no cover - construction
        model = {a: True for a in atoms}
        premises = [("atom", "a")]
        cands = [("atom", "b"), ("atom", "a"), ("not", ("atom", "c")), ("not", ("atom", "d"))]
        model["c"] = model["d"] = True

    shown, label_of = _label_items(rng, cands)
    obs = Rec(structure=Lst([Pred("value", Ident(a), Ident("true" if model[a] else "false"))
                             for a in atoms]),
              premises=Lst([_fsym(p) for p in premises]),
              claims=Lst([Pred("claim", Ident(lab), _fsym(f)) for lab, f in shown]),
              query=Pred("conjecture", Pred("true_in_structure_and_not_implied")))
    return obs, _shuffled(rng, [lab for lab, _ in shown]), label_of[0], {
        "model": {k: bool(v) for k, v in model.items()},
        "answer_claim": str(_fsym(cands[0]))}


class ConjectureGeneration(Lesson):
    """A claim both true in the structure and not already implied."""

    id = "conjecture_generation"
    level = 82
    tags = ("mathematics", "formal-reasoning")
    teaches = "a claim both true in the structure and not already implied"
    capabilities = ('model_checking', 'entailment', 'informativeness')
    axes = {'reasoning_depth': 4, 'compositional_depth': 3, 'world_complexity': 3}

    generate = staticmethod(gen_conjecture_generation)
