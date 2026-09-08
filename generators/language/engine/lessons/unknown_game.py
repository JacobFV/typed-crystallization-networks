"""``unknown_game`` — infer what is relevant, what moves do, and what payoff means.

Ultimate transfer and open-world capstones.
"""

from __future__ import annotations

import itertools
import random
from typing import Sequence

from .._structure import Ident, Lst, Num, Pred, Rec, Term
from ..lesson import Lesson
from ..generators.capstone import _nonce_pool, _shuffled


def gen_unknown_game(rng: random.Random, ctx):
    """A game with no semantic priors at all: observations are records of nonce
    feature tokens, actions are nonce tokens, and payoffs are nonce outcome
    tokens ordered by a preference chain that is itself given only as adjacent
    edges.

    Exactly one observation dimension matters. The other dimensions are live
    distractors, not noise: the learner has to *rule them out*, and the
    generator only emits an episode when every irrelevant dimension is actually
    contradicted somewhere in the log (two rows agreeing on that dimension and
    on the action but differing in payoff). The queried observation never
    occurred verbatim, so row lookup fails, and every (relevant value, move)
    cell does occur, so the answer is uniquely determined by the log.

    Two design choices exist purely to kill a shortcut that otherwise buys a
    third of the episodes without any inference:

    * the payoff table is **anti-symmetric** in the relevant feature (the ranks
      under its two values sum to a constant for every move), so no move is
      better marginally and payoff-averaging over the whole log is worthless;
    * the log is a **balanced design** — its rows are closed under flipping the
      relevant dimension and one decoy dimension — so for any *wrong* dimension,
      the rows matching the query on that dimension contain exactly as many
      rows of each relevant-feature value, and averaging under a wrong dimension
      comes out an exact tie rather than a noisy near-tie. Sampling noise in an
      unbalanced log is worth ~0.45 accuracy on its own; here it is worth
      nothing, and the only remaining route is the intended one.
    """
    n_dims = rng.randint(*ctx.span((4, 5), (7, 8)))   # decoy dimensions to rule out
    n_acts = 3
    n_out = 3
    pool = _nonce_pool(rng, n_dims * 2 + n_acts + n_out)
    rng.shuffle(pool)
    vals = [[pool[2 * d], pool[2 * d + 1]] for d in range(n_dims)]
    a_tok = pool[2 * n_dims:2 * n_dims + n_acts]
    o_tok = pool[2 * n_dims + n_acts:]
    pref = _shuffled(rng, o_tok)
    rank = {t: i for i, t in enumerate(pref)}         # 0 = best

    chosen = None
    for _ in range(500):
        d_star, d_decoy = rng.sample(range(n_dims), 2)
        perm = _shuffled(rng, list(range(n_acts)))
        table = {}
        for a in range(n_acts):
            table[(0, a)] = pref[perm[a]]
            table[(1, a)] = pref[n_acts - 1 - perm[a]]
        free = [d for d in range(n_dims) if d not in (d_star, d_decoy)]
        base = _shuffled(rng, list(itertools.product((0, 1), repeat=len(free))))
        base = base[:rng.choice([2, 3])]
        rows = []
        for a in range(n_acts):                       # the balanced design
            for b in base:
                for f_star in (0, 1):
                    for f_dec in (0, 1):
                        ob = [0] * n_dims
                        for d, v in zip(free, b):
                            ob[d] = v
                        ob[d_star], ob[d_decoy] = f_star, f_dec
                        rows.append((tuple(ob), a, table[(f_star, a)]))
        rows = _shuffled(rng, rows)

        # identifiability: every other dimension must be refuted by the log
        def consistent(d: int) -> bool:
            seen: dict[tuple[int, int], str] = {}
            for ob, a, o in rows:
                key = (ob[d], a)
                if key in seen and seen[key] != o:
                    return False
                seen[key] = o
            return True
        if any(consistent(d) for d in range(n_dims) if d != d_star):
            continue
        seen_obs = {ob for ob, _a, _o in rows}
        fresh = [q for q in itertools.product((0, 1), repeat=n_dims) if q not in seen_obs]
        if not fresh:
            continue
        q_ob = list(fresh[rng.randrange(len(fresh))])
        best_a = min(range(n_acts), key=lambda a: rank[table[(q_ob[d_star], a)]])
        chosen = (d_star, table, rows, q_ob, best_a)
        break
    if chosen is None:                                # pragma: no cover - construction
        return gen_unknown_game(random.Random(rng.random()), ctx)
    d_star, table, rows, q_ob, best_a = chosen

    def obs_rec(ob: Sequence[int]) -> Term:
        return Lst([Pred("f", Num(d), Ident(vals[d][ob[d]])) for d in range(n_dims)])

    log = [Pred("round", obs_rec(ob), Ident(a_tok[a]), Ident(o)) for ob, a, o in rows]
    edges = [Pred("better", Ident(pref[i]), Ident(pref[i + 1])) for i in range(n_out - 1)]
    obs = Rec(log=Lst(log),
              signals=Lst(_shuffled(rng, edges)),
              interface=Lst([Pred("entry", Ident("state"), Ident("move"), Ident("payoff"))]),
              now=obs_rec(q_ob),
              query=Ident("move_with_best_payoff"))
    hidden = {"relevant_dimension": d_star, "answer": a_tok[best_a], "preference": pref,
              "payoffs": {f"{vals[d_star][v]}|{a_tok[a]}": table[(v, a)]
                          for v in (0, 1) for a in range(n_acts)},
              "n_dims": n_dims, "n_rows": len(rows), "query_feature": vals[d_star][q_ob[d_star]]}
    return obs, _shuffled(rng, a_tok), a_tok[best_a], hidden


class UnknownGame(Lesson):
    """Infer what is relevant, what moves do, and what payoff means."""

    id = "unknown_game"
    level = 168
    tags = ("transfer", "capstone", "open-world")
    teaches = "infer what is relevant, what moves do, and what payoff means"
    capabilities = ('ontology_learning', 'scientific_induction', 'open_ended_discovery', 'abstraction')
    axes = {'lexical_novelty': 5, 'world_complexity': 4, 'reasoning_depth': 5, 'ambiguity': 5}

    generate = staticmethod(gen_unknown_game)
