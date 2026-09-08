"""``civilization_simulator`` — multi-generation cultural transmission under a stated rule.

Civilization-scale symbolic learning.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.capstone import _mode, _nonce_pool, _shuffled, _transmit


def gen_civilization_simulator(rng: random.Random, ctx):
    """A ring of agents, each holding a convention, transmitting for G
    generations under one of two explicitly stated rules; which convention ends
    up dominant?

    The rule (``copy the majority of your neighbourhood`` vs ``copy the most
    prestigious member of your neighbourhood``) is drawn per episode and written
    into the observation, so the dynamics have to be read, not assumed. Ring
    adjacency is given as facts over shuffled agent ids, so list order is not
    the ring. Episodes are rejection-sampled so that ``answer == the initial
    plurality`` happens at the chance rate ``1/k``: counting the starting
    distribution, the obvious shortcut, is worth 0.29-0.37 depending on ``k``
    against 0.25-0.33 chance — the residue is the minority of worlds whose
    dynamics simply cannot be made to overturn the starting plurality.
    """
    n = rng.randint(*ctx.span((7, 9), (15, 19)))
    k = rng.randint(3, 4)
    words = _nonce_pool(rng, k)
    rule = rng.choice(["majority", "prestige"])
    gens = rng.randint(2, 4)
    want_same = rng.random() < 1.0 / k               # heuristic accuracy -> 1/k
    fallback = None
    for _ in range(600):
        start = [rng.choice(words) for _ in range(n)]
        if len({*start}) < 2:
            continue
        prestige = rng.sample(range(10, 90), n)
        state = list(start)
        for _ in range(gens):
            state = _transmit(state, prestige, rule)
        answer, strict = _mode(state)
        if not strict:
            continue
        init_top, init_strict = _mode(start)
        if not init_strict:
            continue
        fallback = (start, prestige, state, answer, init_top)
        if (answer == init_top) == want_same:
            break
    if fallback is None:                              # pragma: no cover - construction
        start = [words[i % k] for i in range(n)]
        prestige = list(range(10, 10 + n))
        state = _transmit(start, prestige, rule)
        answer, init_top = _mode(state)[0], _mode(start)[0]
        fallback = (start, prestige, state, answer, init_top)
    start, prestige, final, answer, init_top = fallback

    ids = _shuffled(rng, [f"g{i}" for i in range(n)])   # ring position != id order
    agents = [Pred("agent", Ident(ids[i]), Ident(start[i]), Num(prestige[i])) for i in range(n)]
    links = [Pred("neighbour", Ident(ids[i]), Ident(ids[(i + 1) % n])) for i in range(n)]
    rules = [Pred("rule", Ident("copy_" + rule)),
             Pred("tiebreak", Pred("highest_prestige")),
             Pred("neighbourhood", Pred("self_and_both_neighbours")),
             Pred("update", Ident("simultaneous")),
             Pred("generations", Num(gens))]
    obs = Rec(population=Lst(_shuffled(rng, agents)),
              structure=Lst(_shuffled(rng, links)),
              dynamics=Lst(rules),
              query=Ident("dominant_convention_after_generations"))
    hidden = {"rule": rule, "generations": gens, "initial": start, "final": final,
              "initial_plurality": init_top, "answer": answer, "n_agents": n}
    return obs, _shuffled(rng, words), answer, hidden


class CivilizationSimulator(Lesson):
    """Multi-generation cultural transmission under a stated rule."""

    id = "civilization_simulator"
    level = 163
    tags = ("civilization-scale",)
    teaches = "multi-generation cultural transmission under a stated rule"
    capabilities = ('multi_agent_coordination', 'planning', 'abstraction')
    axes = {'world_complexity': 5, 'reasoning_depth': 4, 'discourse_horizon': 4, 'lexical_novelty': 3}

    generate = staticmethod(gen_civilization_simulator)
