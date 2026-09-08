"""``default_reasoning`` — defeasible rules, exceptions, and silence.

Mathematics and formal reasoning.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.mathematics import _nonce, _nonces, _shuffled


def gen_default_reasoning(rng: random.Random, ctx):
    """Birds fly, penguins do not, and nobody said anything about rocks.

    A nonce taxonomy carries defeasible rules at several levels; the conclusion
    is fixed by the *most specific* rule above the individual, and by nothing at
    all when no rule applies — so the three answers (yes / no / unknown) each
    require reading a different part of the hierarchy. The target answer is
    drawn uniformly and the taxonomy is then built to realize it, with an
    opposite-polarity rule placed higher up whenever specificity is what decides
    the case."""
    depth = ctx.at(4, 10, default=4)
    classes = _nonces(rng, depth, 2)
    prop, other = _nonces(rng, 2, 2, avoid=classes)
    ind = _nonce(rng, 2)
    want = rng.choice(["yes", "no", "unknown"])
    # an "unknown" episode still shows defaults — they just sit *below* the
    # individual's class, so the level it sits at is drawn to leave room for them
    j = rng.randint(1, depth - 2) if want == "unknown" else rng.randint(1, depth - 1)

    rules: list[tuple[str, str, bool]] = []              # (class, property, polarity)
    if want == "unknown":
        for m in range(j + 1, depth):                    # rules exist, but only below
            rules.append((classes[m], prop, rng.random() < 0.5))
    else:
        m = rng.randint(0, j)                            # nearest applicable rule
        pol = want == "yes"
        rules.append((classes[m], prop, pol))
        if m > 0 and rng.random() < 0.8:                 # a general rule it overrides
            rules.append((classes[rng.randrange(m)], prop, not pol))
        for mm in range(j + 1, depth):                   # irrelevant, more specific
            if rng.random() < 0.5:
                rules.append((classes[mm], prop, rng.random() < 0.5))
    for _ in range(rng.randint(1, 2)):                   # distractor property
        rules.append((classes[rng.randrange(depth)], other, rng.random() < 0.5))

    # recompute the answer from the constructed knowledge base
    answer = "unknown"
    for m in range(j, -1, -1):
        hits = [pol for cl, pr, pol in rules if cl == classes[m] and pr == prop]
        if hits:
            answer = "yes" if hits[0] else "no"
            break

    taxonomy = [Pred("subclass", Ident(classes[i + 1]), Ident(classes[i])) for i in range(depth - 1)]
    defaults = [Pred("usually" if pol else "usually_not", Ident(cl), Ident(pr))
                for cl, pr, pol in rules]
    obs = Rec(taxonomy=Lst(_shuffled(rng, taxonomy)),
              defaults=Lst(_shuffled(rng, defaults)),
              facts=Lst([Pred("member", Ident(ind), Ident(classes[j]))]),
              options=Lst([Ident("yes"), Ident("no"), Ident("unknown")]),
              query=Pred("holds", Ident(prop), Ident(ind)))
    return obs, _shuffled(rng, ["yes", "no", "unknown"]), answer, {
        "level": j, "property": prop, "answer": answer, "n_defaults": len(rules)}


class DefaultReasoning(Lesson):
    """Defeasible rules, exceptions, and silence."""

    id = "default_reasoning"
    level = 91
    tags = ("mathematics", "formal-reasoning")
    teaches = "defeasible rules, exceptions, and silence"
    capabilities = ('nonmonotonic', 'specificity', 'taxonomy')
    axes = {'reasoning_depth': 4, 'world_complexity': 3, 'ambiguity': 3}
    answers = ['yes', 'no', 'unknown']

    generate = staticmethod(gen_default_reasoning)
