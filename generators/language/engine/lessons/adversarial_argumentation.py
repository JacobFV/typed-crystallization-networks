"""``adversarial_argumentation`` — which thesis survives valid criticism.

Epistemics, argument, and teaching.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.epistemics import _grounded, _nonces, _shuffled


def gen_adversarial_argumentation(rng: random.Random, ctx):
    """A critic attacks every thesis; exactly one survives valid criticism.

    Each of the four theses is attacked. One is defended by an argument the
    critic never answered (so it is IN); the others are either undefended, or
    defended by a rebuttal that is itself successfully rebutted, or caught in an
    attack cycle that leaves them merely undecided. Surviving is computed on the
    graph, so "the thesis with the most defenders" is not the answer.
    """
    n = ctx.at(4, 9, default=4)
    for _ in range(200):
        names = _nonces(rng, 4 * n, 4)
        theses = names[:n]
        attackers = names[n:2 * n]
        defenders = names[2 * n:3 * n]
        extra = names[3 * n:]
        winner = rng.randrange(n)
        attacks = [(attackers[i], theses[i]) for i in range(n)]
        for i in range(n):
            if i == winner:
                attacks.append((defenders[i], attackers[i]))       # unanswered defence
            else:
                kind = rng.choice(["none", "rebutted", "cycle"])
                if kind == "rebutted":
                    attacks.append((defenders[i], attackers[i]))
                    attacks.append((extra[i % len(extra)], defenders[i]))
                elif kind == "cycle":
                    attacks.append((defenders[i], attackers[i]))
                    attacks.append((attackers[i], defenders[i]))
        nodes = names
        inn, out, und = _grounded(nodes, attacks)
        survivors = [t for t in theses if t in inn]
        if len(survivors) == 1:
            break
    else:                                     # pragma: no cover - construction
        survivors = [theses[winner]]

    obs = Rec(theses=Lst([Pred("thesis", Ident(t)) for t in _shuffled(rng, theses)]),
              arguments=Lst([Pred("argument", Ident(x)) for x in _shuffled(rng, names[n:])]),
              attacks=Lst(_shuffled(rng, [Pred("attacks", Ident(a), Ident(b)) for a, b in attacks])),
              query=Ident("which_thesis_survives"))
    return (obs, _shuffled(rng, theses), survivors[0],
            {"survivor": survivors[0], "n_attacks": len(attacks),
             "undecided": sorted(t for t in theses if t in und)})


class AdversarialArgumentation(Lesson):
    """Which thesis survives valid criticism."""

    id = "adversarial_argumentation"
    level = 97
    tags = ("epistemics", "argument", "teaching")
    teaches = "which thesis survives valid criticism"
    capabilities = ('argumentation', 'defeasible_reasoning', 'adversariality')
    axes = {'reasoning_depth': 4, 'adversariality': 3, 'world_complexity': 4}

    generate = staticmethod(gen_adversarial_argumentation)
