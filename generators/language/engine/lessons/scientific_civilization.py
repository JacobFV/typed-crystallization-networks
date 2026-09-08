"""``scientific_civilization`` — cumulative consensus from partial, diffusing evidence.

Civilization-scale symbolic learning.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.capstone import _adopt, _mode, _nonce_pool, _shuffled


def gen_scientific_civilization(rng: random.Random, ctx):
    """Labs hold theories, know only part of the experimental record, and share
    what they know along a collaboration network; after R rounds of sharing and
    re-adoption, which theory is the consensus?

    Cumulative epistemic progress is bounded here by *who has seen what*: a lab
    re-adopts on its own evidence, so the consensus is a property of the
    diffusion path, not of the global record. Episodes are rejection-sampled so
    that the globally best-supported theory is the consensus only at chance rate
    (measured 0.34 against 0.33) — the lesson is the simulation, not the tally.
    """
    n_th = 3
    n_labs = rng.randint(*ctx.span((4, 5), (7, 8)))
    n_exp = rng.randint(5, 6)
    rounds = rng.randint(*ctx.span((1, 2), (3, 4)))
    pool = _nonce_pool(rng, n_th + n_labs + n_exp + 2)
    theories, pool = pool[:n_th], pool[n_th:]
    labs, pool = pool[:n_labs], pool[n_labs:]
    exps, pool = pool[:n_exp], pool[n_exp:]
    o_pos, o_neg = pool[0], pool[1]
    want_same = rng.random() < 1.0 / n_th
    fallback = None

    for _ in range(800):
        results = {e: rng.choice([o_pos, o_neg]) for e in exps}
        preds = {t: {e: rng.choice([o_pos, o_neg]) for e in exps} for t in theories}
        edges = [(a, b) for i, a in enumerate(labs) for b in labs[i + 1:] if rng.random() < 0.5]
        if not edges:
            continue
        known = {l: set(rng.sample(exps, 2)) for l in labs}
        held = {l: theories[i % n_th] for i, l in enumerate(_shuffled(rng, labs))}
        state = dict(held)
        know = {l: set(v) for l, v in known.items()}
        for _ in range(rounds):
            grown = {l: set(know[l]) for l in labs}
            for a, b in edges:
                grown[a] |= know[b]
                grown[b] |= know[a]
            know = grown
            state = {l: _adopt(sorted(know[l]), state[l], theories, preds, results) for l in labs}
        answer, strict = _mode([state[l] for l in labs])
        if not strict:
            continue
        glob = {t: sum(1 for e in exps if preds[t][e] == results[e]) for t in theories}
        gbest = max(glob.values())
        if sum(1 for t in theories if glob[t] == gbest) != 1:
            continue
        global_best = [t for t in theories if glob[t] == gbest][0]
        fallback = (results, preds, edges, known, held, state, answer, global_best, glob)
        if (answer == global_best) == want_same:
            break
    if fallback is None:                              # pragma: no cover - construction
        results = {e: o_pos for e in exps}
        preds = {t: {e: (o_pos if i == 0 else o_neg) for e in exps} for i, t in enumerate(theories)}
        edges = [(labs[0], labs[1])]
        known = {l: {exps[0]} for l in labs}
        held = {l: theories[0] for l in labs}
        state = dict(held)
        fallback = (results, preds, edges, known, held, state, theories[0], theories[0], {})
    results, preds, edges, known, held, state, answer, global_best, glob = fallback

    record = [Pred("result", Ident(e), Ident(results[e])) for e in exps]
    claims = [Pred("predicts", Ident(t), Ident(e), Ident(preds[t][e])) for t in theories for e in exps]
    holds = [Pred("holds", Ident(l), Ident(held[l])) for l in labs]
    seen = [Pred("has_seen", Ident(l), Ident(e)) for l in labs for e in sorted(known[l])]
    links = [Pred("shares_with", Ident(a), Ident(b)) for a, b in edges]
    protocol = [Pred("round", Pred("share_then_readopt")),
                Pred("adopt", Pred("most_supported_by_own_evidence")),
                Pred("tiebreak", Pred("keep_current_else_first_listed")),
                Pred("rounds", Num(rounds))]
    obs = Rec(theories=Lst([Ident(t) for t in theories]),      # listed order = the tiebreak order
              record=Lst(_shuffled(rng, record)),
              claims=Lst(_shuffled(rng, claims)),
              labs=Lst(_shuffled(rng, holds)),
              evidence=Lst(_shuffled(rng, seen)),
              network=Lst(_shuffled(rng, links)),
              protocol=Lst(protocol),
              query=Ident("consensus_theory_after_rounds"))
    hidden = {"rounds": rounds, "final": {l: state[l] for l in labs}, "answer": answer,
              "global_best": global_best, "global_scores": glob, "n_labs": len(labs)}
    return obs, _shuffled(rng, theories), answer, hidden


class ScientificCivilization(Lesson):
    """Cumulative consensus from partial, diffusing evidence."""

    id = "scientific_civilization"
    level = 164
    tags = ("civilization-scale",)
    teaches = "cumulative consensus from partial, diffusing evidence"
    capabilities = ('scientific_induction', 'multi_agent_coordination', 'belief_modeling')
    axes = {'world_complexity': 5, 'reasoning_depth': 5, 'discourse_horizon': 4, 'lexical_novelty': 3, 'ambiguity': 2}

    generate = staticmethod(gen_scientific_civilization)
