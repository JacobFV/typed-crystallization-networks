"""``argumentation`` — grounded status of a claim in an attack graph.

Epistemics, argument, and teaching.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.epistemics import ARG_LABELS, _grounded, _nonces, _shuffled


def gen_argumentation(rng: random.Random, ctx):
    """Claims and attacks form a graph; status is the grounded extension.

    Attack edges are sampled with cycles allowed, so all three statuses actually
    occur, and the target status is drawn first and the graph resampled until it
    realizes it — a claim is not accepted because it is attacked least, but
    because its attackers are themselves defeated.
    """
    target = rng.choice(ARG_LABELS)
    for _ in range(400):
        n = rng.randint(*ctx.span((5, 7), (10, 14)))
        nodes = _nonces(rng, n, 4)
        attacks: list[tuple[str, str]] = []
        for a in nodes:
            for b in nodes:
                if a != b and rng.random() < 0.22:
                    attacks.append((a, b))
        inn, out, und = _grounded(nodes, attacks)
        pool = {"accepted": sorted(inn), "rejected": sorted(out), "undecided": sorted(und)}[target]
        if pool:
            q = rng.choice(pool)
            break
    else:                                     # pragma: no cover - construction
        nodes, attacks = _nonces(rng, 2, 4), []
        inn, out, und = _grounded(nodes, attacks)
        q, target = nodes[0], "accepted"

    obs = Rec(claims=Lst([Pred("claim", Ident(x)) for x in _shuffled(rng, nodes)]),
              attacks=Lst(_shuffled(rng, [Pred("attacks", Ident(a), Ident(b)) for a, b in attacks])),
              query=Pred("status_of", Ident(q)))
    return (obs, _shuffled(rng, ARG_LABELS), target,
            {"claim": q, "label": target, "n_claims": len(nodes), "n_attacks": len(attacks),
             "accepted": sorted(inn), "undecided": sorted(und)})


class Argumentation(Lesson):
    """Grounded status of a claim in an attack graph."""

    id = "argumentation"
    level = 96
    tags = ("epistemics", "argument", "teaching")
    teaches = "grounded status of a claim in an attack graph"
    capabilities = ('argumentation', 'defeasible_reasoning', 'graph_reasoning')
    axes = {'reasoning_depth': 4, 'recursion_depth': 3, 'world_complexity': 3}
    answers = ['accepted', 'rejected', 'undecided']

    generate = staticmethod(gen_argumentation)
