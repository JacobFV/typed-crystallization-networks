"""``predicate_logic`` — facts + rules -> derived bindings.

Compositional semantics and logical language.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.base import NAMES
from ..generators.semantics import BASE_KINDS, DERIVED_KINDS, _shuffled


def gen_predicate_logic(rng: random.Random, ctx):
    """Facts + universal rules; answer the *binding* that satisfies the goal.

    A two-step chain ``p(X) -> m(X)``, ``m(X) -> g(X)`` is asserted for exactly
    one entity, and decoy rules fire for the others, so neither "the entity with
    a fact" nor "the entity named in a rule" is the answer.
    """
    depth = ctx.at(2, 5, default=2)               # rule applications between fact and goal
    ents = rng.sample(NAMES, 4)
    trigger, decoy_a, decoy_b = rng.sample(BASE_KINDS, 3)
    dkinds = rng.sample(DERIVED_KINDS, depth + 1)
    chain, other = dkinds[:depth], dkinds[depth]
    goal = chain[-1]
    target = rng.choice(ents)

    facts = [Pred("fact", Ident(trigger), Ident(target))]
    for e in ents:
        if e == target:
            continue
        facts.append(Pred("fact", Ident(rng.choice([decoy_a, decoy_b])), Ident(e)))
    rules = [Pred("rule", Ident(trigger), Ident(chain[0]))]
    rules += [Pred("rule", Ident(chain[i]), Ident(chain[i + 1])) for i in range(depth - 1)]
    rules.append(Pred("rule", Ident(decoy_a), Ident(other)))
    rng.shuffle(facts)
    rng.shuffle(rules)

    # forward chaining, so the answer is derived rather than asserted
    kb = {(str(f.value[1].value), str(f.value[2].value)) for f in facts}
    changed = True
    while changed:
        changed = False
        for r in rules:
            a, b = str(r.value[1].value), str(r.value[2].value)
            for (p, e) in sorted(kb):
                if p == a and (b, e) not in kb:
                    kb.add((b, e))
                    changed = True
    derived = sorted(e for (p, e) in kb if p == goal)
    if len(derived) != 1:                                  # never happens by construction
        return gen_predicate_logic(random.Random(rng.random()), ctx)

    obs = Rec(facts=Lst(facts), rules=Lst(rules), entities=Lst([Ident(e) for e in ents]),
              query=Pred("who", Ident(goal)))
    return obs, _shuffled(rng, ents), derived[0], {"goal": goal, "chain": [trigger, *chain],
                                                   "target": target}


class PredicateLogic(Lesson):
    """Facts + rules -> derived bindings."""

    id = "predicate_logic"
    level = 12
    tags = ("compositional-semantics", "logic")
    teaches = "facts + rules -> derived bindings"
    capabilities = ('proof_search', 'variable_binding', 'abstraction')
    axes = {'reasoning_depth': 3, 'compositional_depth': 3, 'world_complexity': 2}

    generate = staticmethod(gen_predicate_logic)
