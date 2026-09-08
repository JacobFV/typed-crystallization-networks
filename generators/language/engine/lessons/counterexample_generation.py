"""``counterexample_generation`` — the small model that refutes a false universal.

Mathematics and formal reasoning.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Rec
from ..lesson import Lesson
from ..generators.mathematics import _CLAIMS, _CLAIM_SYM, _label_items, _model_sym, _rand_model, _shuffled


def gen_counterexample_generation(rng: random.Random, ctx):
    """A universal claim that is false, and the small model that refutes it.

    Four three-element structures are drawn and the claim is *evaluated* in each;
    the episode survives only when exactly one of them is a countermodel, so
    picking the refuter requires checking the quantifiers rather than pattern
    matching on how the claim is written."""
    dom = ["e1", "e2", "e3"]
    n_models = ctx.at(4, 8, default=4)
    for _ in range(400):
        name, test = rng.choice(_CLAIMS)
        models = [_rand_model(rng, dom) for _ in range(n_models)]
        verdicts = [test(dom, *m) for m in models]
        if verdicts.count(False) != 1:
            continue
        i = verdicts.index(False)
        order = [models[i]] + [m for j, m in enumerate(models) if j != i]
        break
    else:                                                # pragma: no cover - construction
        name, test = _CLAIMS[4]
        bad = ({x: False for x in dom}, {x: False for x in dom},
               {(x, y): (x == y) for x in dom for y in dom})
        good = ({x: False for x in dom}, {x: False for x in dom},
                {(x, y): False for x in dom for y in dom})
        order = [bad, good, good, good]

    shown, label_of = _label_items(rng, order, prefix="m")
    obs = Rec(claim=_CLAIM_SYM[name],
              models=Lst([_model_sym(lab, dom, *m) for lab, m in shown]),
              query=Ident("model_that_refutes_the_claim"))
    return obs, _shuffled(rng, [lab for lab, _ in shown]), label_of[0], {
        "claim": name, "domain_size": len(dom)}


class CounterexampleGeneration(Lesson):
    """The small model that refutes a false universal."""

    id = "counterexample_generation"
    level = 87
    tags = ("mathematics", "formal-reasoning")
    teaches = "the small model that refutes a false universal"
    capabilities = ('model_checking', 'quantification', 'falsification')
    axes = {'reasoning_depth': 4, 'world_complexity': 3, 'compositional_depth': 3}

    generate = staticmethod(gen_counterexample_generation)
