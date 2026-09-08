"""``universal_interface_transfer`` — bootstrap states, actions and reward semantics from a trace, then plan.

Ultimate transfer and open-world capstones.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec, Term
from ..lesson import Lesson
from ..generators.capstone import _bfs_dist, _nonce_pool, _shuffled


def gen_universal_interface_transfer(rng: random.Random, ctx):
    """An environment whose entire language is permuted: state tokens, action
    tokens and outcome tokens are drawn from **one shared pool of fresh nonce
    words**, so even the type of a symbol has to be inferred from where it
    occurs. The learner sees an interaction log and a preference chain over
    outcome symbols, and must say which action, taken now, starts the shortest
    route to the best outcome.

    Four things must be bootstrapped from the trace alone: which tokens are
    states, what each action token does (the transition table), which outcome
    token is best (the chain is given only as adjacent edges, so it must be
    closed transitively), and then a 2-3 step plan. The generator guarantees
    identifiability: the log covers *every* (state, action) pair, the goal is at
    distance 2 or 3, and exactly one action attains that distance — episodes
    that fail any of these are discarded rather than graded. Every action taken
    from the current state leads to the same signal, so one-step greed is
    deliberately worthless.
    """
    n_states = rng.randint(*ctx.span((4, 5), (9, 11)))
    n_acts = rng.randint(3, 4)
    n_out = 3
    pool = _nonce_pool(rng, n_states + n_acts + n_out)
    rng.shuffle(pool)
    s_tok = pool[:n_states]
    a_tok = pool[n_states:n_states + n_acts]
    o_tok = pool[n_states + n_acts:]
    pref = _shuffled(rng, o_tok)                     # pref[0] strictly best

    chosen = None
    for _ in range(400):
        delta = {(s, a): rng.randrange(n_states) for s in range(n_states) for a in range(n_acts)}
        goal = rng.randrange(n_states)
        outcome = [rng.choice(pref[1:]) for _ in range(n_states)]
        outcome[goal] = pref[0]
        dist = _bfs_dist(delta, n_states, n_acts, goal)
        cur_opts = []
        for s in range(n_states):
            if not 2 <= dist[s] <= 3:
                continue
            best = [a for a in range(n_acts) if dist[delta[(s, a)]] == dist[s] - 1]
            if len(best) == 1:
                cur_opts.append((s, best[0]))
        if cur_opts:
            cur, act = cur_opts[rng.randrange(len(cur_opts))]
            chosen = (delta, goal, outcome, dist, cur, act)
            break
    if chosen is None:                                # pragma: no cover - construction
        return gen_universal_interface_transfer(random.Random(rng.random()))
    delta, goal, outcome, dist, cur, act = chosen
    # every action available *now* leads to the same signal, so "take the action
    # with the nicest immediate signal" is worth exactly nothing and the episode
    # can only be answered by planning through the inferred transition table.
    # Safe because the goal is two or three steps away: no successor is the goal.
    for a in range(n_acts):
        outcome[delta[(cur, a)]] = pref[1]

    # a log of short rollouts, extended until every (state, action) pair appears:
    # without full coverage the transition table is not identifiable from the
    # trace, and the question would be partly unanswerable rather than hard
    steps: list[Term] = []
    covered: set[tuple[int, int]] = set()
    for _ in range(40):
        s = rng.randrange(n_states)
        for _ in range(rng.randint(3, 5)):
            a = rng.randrange(n_acts)
            t = delta[(s, a)]
            steps.append(Pred("t", Ident(s_tok[s]), Ident(a_tok[a]),
                              Ident(s_tok[t]), Ident(outcome[t])))
            covered.add((s, a))
            s = t
        if len(covered) == n_states * n_acts:
            break
    for s in range(n_states):                         # top up anything still missing
        for a in range(n_acts):
            if (s, a) in covered:
                continue
            t = delta[(s, a)]
            steps.append(Pred("t", Ident(s_tok[s]), Ident(a_tok[a]),
                              Ident(s_tok[t]), Ident(outcome[t])))
            covered.add((s, a))

    edges = [Pred("better", Ident(pref[i]), Ident(pref[i + 1])) for i in range(n_out - 1)]
    obs = Rec(log=Lst(steps),
              signals=Lst(_shuffled(rng, edges)),
              interface=Lst([Pred("entry", Ident("from"), Ident("act"), Ident("to"), Ident("signal"))]),
              now=Ident(s_tok[cur]),
              query=Ident("act_toward_best_signal"))
    hidden = {"states": s_tok, "actions": a_tok, "preference": pref, "goal_state": s_tok[goal],
              "current": s_tok[cur], "distance": dist[cur], "answer": a_tok[act],
              "transitions": {f"{s_tok[s]}|{a_tok[a]}": s_tok[delta[(s, a)]]
                              for s in range(n_states) for a in range(n_acts)}}
    return obs, _shuffled(rng, a_tok), a_tok[act], hidden


class UniversalInterfaceTransfer(Lesson):
    """Bootstrap states, actions and reward semantics from a trace, then plan."""

    id = "universal_interface_transfer"
    level = 167
    tags = ("transfer", "capstone", "open-world")
    teaches = "bootstrap states, actions and reward semantics from a trace, then plan"
    capabilities = ('ontology_learning', 'planning', 'open_ended_discovery', 'finite_state_induction')
    axes = {'lexical_novelty': 5, 'world_complexity': 4, 'reasoning_depth': 5, 'ambiguity': 4, 'discourse_horizon': 4}

    generate = staticmethod(gen_universal_interface_transfer)
