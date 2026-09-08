"""``strategy_transfer`` — the same strategy under wholly renamed symbols.

Problem formulation and hierarchical agency.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec, Term
from ..lesson import Lesson
from ..generators.epistemics import _game_draw, _game_pools, _nonces, _shuffled


def gen_strategy_transfer(rng: random.Random, ctx):
    """The same strategy, with every symbol replaced and magnitudes re-encoded.

    Nothing about the surface survives from :func:`gen_strategy_discovery`: the
    predicates, the resource, the move names and the option for "no winning
    move" are all fresh nonce words per episode, and quantities appear as tally
    lists rather than numerals. The game is structurally identical, so a learner
    that induced the invariant transfers and a learner that memorized surface
    forms — including the token ``take_2`` — has nothing to carry.
    """
    hi = ctx.at(16, 48, default=16)
    for _ in range(200):
        moves = sorted(rng.sample([1, 2, 3, 4, 5, 6][:ctx.at(4, 6, default=4)],
                                  rng.randint(*ctx.span((2, 3), (4, 5)))))
        words = _nonces(rng, 6 + len(moves), 4)
        heap_pred, move_pred, rule_pred, inst_pred, mark, hold = words[:6]
        move_ids = words[6:]
        naming = dict(zip(moves, move_ids))
        pools = _game_pools(moves, hi, hold, naming)
        if len(pools) >= 3:
            break
    shown = []
    used: list[int] = []
    for _ in range(4):
        k, lab = _game_draw(rng, pools, used)
        used.append(k)
        shown.append((k, lab))
    q, answer = _game_draw(rng, pools, used)
    vocab = list(move_ids) + [hold]

    def tally(k: int) -> Term:
        return Lst([Ident(mark) for _ in range(k)])

    obs = Rec(moves=Lst(_shuffled(rng, [Pred(move_pred, Ident(naming[m]), tally(m))
                                        for m in moves])),
              rule=Pred(rule_pred, Ident(hold), Ident(heap_pred)),
              solved=Lst(_shuffled(rng, [Pred(inst_pred, tally(k), Ident(a)) for k, a in shown])),
              query=Pred(heap_pred, tally(q)))
    return (obs, _shuffled(rng, vocab), answer,
            {"moves": moves, "position": q, "answer": answer,
             "naming": {str(m): naming[m] for m in moves}, "pass_token": hold})


class StrategyTransfer(Lesson):
    """The same strategy under wholly renamed symbols."""

    id = "strategy_transfer"
    level = 112
    tags = ("problem-formulation", "hierarchical-agency")
    teaches = "the same strategy under wholly renamed symbols"
    capabilities = ('strategy_induction', 'transfer', 'abstraction')
    axes = {'lexical_novelty': 5, 'reasoning_depth': 5, 'ontology_novelty': 4}

    generate = staticmethod(gen_strategy_transfer)
