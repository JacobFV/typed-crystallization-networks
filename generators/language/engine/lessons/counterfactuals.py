"""``counterfactuals`` — re-running a structural causal model under a different intervention.

Analogy, causality, planning, and programs.
"""

from __future__ import annotations

import random
from typing import Mapping

from .._structure import Ident, Lst, Num, Pred, Rec, Str
from ..lesson import Lesson
from ..generators.causal import _nonce_names


def gen_counterfactuals(rng: random.Random, ctx):
    """Given a factual observation and the model, what would the target have
    been under a different intervention?

    The exogenous values are read off the factual world and held fixed while the
    model is re-run with the antecedent forced — the answer is the value that
    re-run produces, computed, not estimated."""
    MOD = 4
    depth = ctx.at(2, 5, default=2)              # generations below the exogenous root
    names = _nonce_names(rng, 1 + 2 * depth)
    rng.shuffle(names)
    R = names[0]
    levels = [[R]] + [names[1 + 2 * k:3 + 2 * k] for k in range(depth)]
    parent = {}
    for k in range(1, depth + 1):                # two children per generation
        for i, v in enumerate(levels[k]):
            parent[v] = levels[k - 1][i] if k > 1 else R
    order = [v for lv in levels for v in lv]
    coef = {v: (rng.choice([1, 3]), rng.randrange(MOD)) for v in parent}   # v = (a*parent + b) % 4
    ancestors = {R: []}
    for v in order[1:]:
        ancestors[v] = [parent[v]] + ancestors[parent[v]]

    def run(root: int, forced: Mapping[str, int] | None = None) -> dict[str, int]:
        forced = forced or {}
        vals: dict[str, int] = {}
        for v in order:
            if v in forced:
                vals[v] = forced[v]
            elif v in parent:
                a, b = coef[v]
                vals[v] = (a * vals[parent[v]] + b) % MOD
            else:
                vals[v] = root
        return vals

    root_val = rng.randrange(MOD)
    factual = run(root_val)
    # antecedent u, target v: v must be a strict descendant of u, else the
    # counterfactual is just the factual value and the model is never used
    pairs = [(u, v) for v in order for u in ancestors[v]]
    u, v = rng.choice(pairs)
    w = rng.choice([x for x in range(MOD) if x != factual[u]])
    answer = run(root_val, forced={u: w})[v]

    eqs = [Pred("equation", Ident(c), Ident(parent[c]), Num(coef[c][0]), Num(coef[c][1]))
           for c in sorted(parent)]
    rng.shuffle(eqs)
    vals = [Pred("value", Ident(x), Num(factual[x])) for x in order]
    rng.shuffle(vals)
    obs = Rec(modulus=Num(MOD),
              form=Str("child = (a * parent + b) mod m"),
              equations=Lst(eqs),
              exogenous=Lst([Pred("root", Ident(R))]),
              observed=Lst(vals),
              query=Pred("counterfactual", Ident(v), Ident(u), Num(w)))
    hidden = {"factual": {k: factual[k] for k in order}, "antecedent": u, "target": v,
              "set_to": w, "root": root_val, "answer": answer}
    return obs, list(range(MOD)), answer, hidden


class Counterfactuals(Lesson):
    """Re-running a structural causal model under a different intervention."""

    id = "counterfactuals"
    level = 43
    tags = ("analogy", "causality", "planning", "programs")
    teaches = "re-running a structural causal model under a different intervention"
    capabilities = ('causal_reasoning', 'variable_binding', 'abstraction')
    axes = {'reasoning_depth': 4, 'world_complexity': 3, 'compositional_depth': 3}
    answers = [0, 1, 2, 3]

    generate = staticmethod(gen_counterfactuals)
