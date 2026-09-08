"""``social_convention_learning`` — arbitrary conventions fixed in-episode.

Analogy, causality, planning, and programs.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.social import ACTIONS, _nonce, _shuffled


def gen_social_convention_learning(rng: random.Random, ctx):
    """An arbitrary signal→action pairing, recoverable only from the episode.

    The convention is a fresh random bijection each episode, so there is nothing
    to memorize across episodes; and the queried signal is never demonstrated
    successfully, so the answer is not a lookup. It follows from the convention
    being *shared and exclusive*: three signals are pinned by successful trials,
    the failed trials rule out more, and one action is left for the fourth signal.
    """
    n_sig = ctx.at(4, 6, default=4)
    signals = [_nonce(rng, 3) for _ in range(n_sig)]
    while len(set(signals)) < n_sig:
        signals = [_nonce(rng, 3) for _ in range(n_sig)]
    actions = rng.sample(ACTIONS, n_sig)
    convention = dict(zip(signals, actions))
    query_ix = rng.randrange(n_sig)
    query_sig = signals[query_ix]

    trials: list[tuple[str, str, bool]] = []
    for i, s in enumerate(signals):
        if i == query_ix:
            continue
        trials.append((s, convention[s], True))                       # pins this signal
        bad = rng.choice([a for a in actions if a != convention[s]])
        trials.append((s, bad, False))
    for _ in range(2):                                                # failures on the query
        bad = rng.choice([a for a in actions if a != convention[query_sig]])
        trials.append((query_sig, bad, False))
    trials = _shuffled(rng, trials)
    obs = Rec(history=Lst([Pred("trial", Num(i), Ident(s), Ident(a), Ident("ok" if ok else "fail"))
                           for i, (s, a, ok) in enumerate(trials)]),
              repertoire=Lst([Ident(a) for a in _shuffled(rng, actions)]),
              query=Pred("act_on", Ident(query_sig)))
    return (obs, _shuffled(rng, actions), convention[query_sig],
            {"convention": dict(convention), "query_signal": query_sig,
             "trials": len(trials)})


class SocialConventionLearning(Lesson):
    """Arbitrary conventions fixed in-episode."""

    id = "social_convention_learning"
    level = 51
    tags = ("analogy", "causality", "planning", "programs")
    teaches = "arbitrary conventions fixed in-episode"
    capabilities = ('multi_agent_coordination', 'ontology_learning')
    axes = {'lexical_novelty': 4, 'reasoning_depth': 3, 'ambiguity': 2}

    generate = staticmethod(gen_social_convention_learning)
