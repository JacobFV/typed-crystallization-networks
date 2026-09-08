"""``norm_reasoning`` — obligation, permission and prohibition under prioritized conflicting norms.

Protocols, institutions, and distributed intelligence.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.reflective import _labels, _shuffled


def gen_norm_reasoning(rng: random.Random, ctx):
    """Obligation, permission or prohibition — under conflicting norms.

    Norms carry conditions and an explicit, total priority order. Several
    applicable norms disagree about the queried action and several inapplicable
    ones (their condition is false in this situation) sit at *higher* priority
    as decoys, so the answer is decided by the highest-priority norm whose
    condition actually holds.
    """
    deontics = ["obligatory", "permitted", "forbidden"]
    conds = ["raining", "night", "emergency", "holiday", "on_duty", "alarm"]
    actions = ["enter", "report", "transfer", "depart"]
    target = rng.choice(deontics)
    for attempt in range(200):
        true_atoms = rng.sample(conds, rng.randint(2, 3))
        false_atoms = [c for c in conds if c not in true_atoms]
        action = rng.choice(actions)
        prios = _shuffled(rng, range(1, ctx.at(10, 24, default=10)))
        norms: list[tuple[str, int, str, str, str]] = []
        p = list(prios)
        n_app = rng.randint(2, 3)
        app_prios = sorted(rng.sample(p, n_app), reverse=True)
        for k in range(n_app):
            d = target if k == 0 else rng.choice([x for x in deontics if x != target])
            cond = rng.choice(true_atoms) if rng.random() < 0.75 else "always"
            norms.append(("", app_prios[k], cond, d, action))
        rest = [x for x in p if x not in app_prios]
        for _ in range(rng.randint(*ctx.span((2, 3), (7, 10)))):   # decoys: false condition, high priority
            pr = max(rest) if rest and rng.random() < 0.6 else rng.choice(rest)
            rest.remove(pr)
            norms.append(("", pr, rng.choice(false_atoms), rng.choice(deontics), action))
        for _ in range(rng.randint(1, 2)):           # decoys: other action
            if not rest:
                break
            pr = rest.pop(rng.randrange(len(rest)))
            other = rng.choice([a for a in actions if a != action])
            norms.append(("", pr, rng.choice(true_atoms + ["always"]), rng.choice(deontics), other))
        nids = _labels(rng, "n", len(norms))
        norms = [(nids[i], *n[1:]) for i, n in enumerate(norms)]
        applicable = [n for n in norms
                      if n[4] == action and (n[2] == "always" or n[2] in true_atoms)]
        if not applicable:
            continue
        answer = max(applicable, key=lambda n: n[1])[3]
        if answer == target or attempt > 150:
            break
    obs = Rec(situation=Lst([Pred("holds", Ident(c)) for c in _shuffled(rng, true_atoms)]),
              norms=Lst(_shuffled(rng, [Pred("norm", Ident(i), Num(pr), Ident(cond),
                                             Ident(d), Ident(act))
                                        for i, pr, cond, d, act in norms])),
              conflict_rule=Pred("resolve_by", Pred("highest_priority_applicable")),
              query=Pred("status_of", Ident(action)))
    return (obs, _shuffled(rng, deontics), answer,
            {"action": action, "status": answer, "n_norms": len(norms),
             "n_applicable": len(applicable)})


class NormReasoning(Lesson):
    """Obligation, permission and prohibition under prioritized conflicting norms."""

    id = "norm_reasoning"
    level = 124
    tags = ("protocols", "institutions", "distributed-intelligence")
    teaches = "obligation, permission and prohibition under prioritized conflicting norms"
    capabilities = ('belief_modeling', 'abstraction', 'metareasoning')
    axes = {'reasoning_depth': 4, 'ambiguity': 3, 'world_complexity': 3}
    answers = ['obligatory', 'permitted', 'forbidden']

    generate = staticmethod(gen_norm_reasoning)
