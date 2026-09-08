"""``mechanism_design`` — which payment rule meets the objective once agents best-respond.

Protocols, institutions, and distributed intelligence.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.base import NAMES
from ..generators.reflective import _labels, _shuffled


def gen_mechanism_design(rng: random.Random, ctx):
    """Which mechanism meets the designer's objective once agents best-respond?

    Two bidders with known values choose between a low and a high bid; each
    candidate mechanism fixes a tie-break and a payment rule. Every mechanism's
    2×2 game is solved by iterated elimination of weakly dominated bids — the
    solution concept is named in the observation, and mechanisms that do not
    settle on one profile are never offered — so "what the agents will do" is a
    fact; the objective (a revenue figure, or efficient allocation) is then
    evaluated at that equilibrium and satisfied by exactly one mechanism.
    """
    fallback = None
    for _ in range(500):
        a, b = rng.sample(NAMES, 2)
        va, vb = rng.sample(range(2, 10), 2)
        values = {a: va, b: vb}
        bids = [2, 6, 4, 8][:ctx.at(2, 4, default=2)]
        specs = []
        for tie in (a, b):
            for kind, k in (("own_bid", 0), ("other_bid", 0), ("fixed", rng.randint(1, 5))):
                specs.append((tie, kind, k))

        def outcome(sp, ba: int, bb: int) -> tuple[str, int]:
            tie, kind, k = sp
            if ba > bb:
                win, other = a, bb
            elif bb > ba:
                win, other = b, ba
            else:
                win, other = tie, ba
            own = ba if win == a else bb
            price = own if kind == "own_bid" else other if kind == "other_bid" else k
            return win, price

        def payoff(sp, ba: int, bb: int, who: str) -> int:
            win, price = outcome(sp, ba, bb)
            return values[who] - price if win == who else 0

        def equilibrium(sp):
            """Iterated elimination of weakly dominated bids; ``None`` if it does
            not settle on a single profile."""
            live = {a: list(bids), b: list(bids)}
            for _ in range(4):
                changed = False
                for who, other in ((a, b), (b, a)):
                    def pay(mine: int, theirs: int, w: str = who) -> int:
                        ba, bb = (mine, theirs) if w == a else (theirs, mine)
                        return payoff(sp, ba, bb, w)
                    for s in list(live[who]):
                        for t in list(live[who]):
                            if s == t or t not in live[who] or len(live[who]) == 1:
                                continue
                            vs = [pay(s, o) for o in live[other]]
                            vt = [pay(t, o) for o in live[other]]
                            if all(x >= y for x, y in zip(vs, vt)) and any(x > y for x, y in zip(vs, vt)):
                                live[who].remove(t)
                                changed = True
                if not changed:
                    break
            if len(live[a]) != 1 or len(live[b]) != 1:
                return None
            return outcome(sp, live[a][0], live[b][0])

        usable = [(sp, equilibrium(sp)) for sp in specs]
        usable = [(sp, e) for sp, e in usable if e is not None]
        if len(usable) < 4:
            continue
        chosen = rng.sample(usable, 4)
        ids = _labels(rng, "m", 4)
        spec = {i: sp for i, (sp, _) in zip(ids, chosen)}
        eq = {i: e for i, (_, e) in zip(ids, chosen)}
        kind = rng.choice(["revenue", "efficient"])
        if kind == "revenue":
            revs = {i: eq[i][1] for i in ids}
            tgt = rng.choice(sorted(set(revs.values())))
            winners = [i for i in ids if revs[i] == tgt]
            objective = Pred("objective", Ident("revenue"), Num(tgt))
        else:
            hi = a if va > vb else b
            winners = [i for i in ids if eq[i][0] == hi]
            tgt = -1
            objective = Pred("objective", Ident("item_to_highest_value"), Num(0))
        cand = (a, b, values, bids, ids, spec, eq, objective, kind, tgt, winners[0] if winners else ids[0])
        if fallback is None and winners:
            fallback = cand
        if len(winners) == 1:
            fallback = cand
            break
    a, b, values, bids, ids, spec, eq, objective, kind, tgt, answer = fallback
    obs = Rec(agents=Lst([Pred("agent", Ident(x), Ident("value"), Num(values[x])) for x in (a, b)]),
              bid_options=Lst([Num(v) for v in bids]),
              allocation=Pred("item_to", Pred("highest_bid")),
              solution_concept=Pred("bids_by", Pred("iterated_weak_dominance")),
              mechanisms=Lst(_shuffled(rng, [Pred("mechanism", Ident(i), Pred("tie_to"),
                                                  Ident(spec[i][0]), Pred("winner_pays"),
                                                  Ident(spec[i][1]), Num(spec[i][2]))
                                             for i in ids])),
              objective=objective,
              query=Ident("which_mechanism"))
    return (obs, _shuffled(rng, ids), answer,
            {"objective": kind, "target": tgt, "equilibria": {i: list(eq[i]) for i in ids},
             "values": values})


class MechanismDesign(Lesson):
    """Which payment rule meets the objective once agents best-respond."""

    id = "mechanism_design"
    level = 126
    tags = ("protocols", "institutions", "distributed-intelligence")
    teaches = "which payment rule meets the objective once agents best-respond"
    capabilities = ('multi_agent_coordination', 'planning', 'metareasoning')
    axes = {'reasoning_depth': 5, 'world_complexity': 4, 'compositional_depth': 3}

    generate = staticmethod(gen_mechanism_design)
