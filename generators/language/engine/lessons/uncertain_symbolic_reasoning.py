"""``uncertain_symbolic_reasoning`` — exact bounds and comparisons over uncertain propositions.

Mathematics and formal reasoning.
"""

from __future__ import annotations

import random
from fractions import Fraction

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.mathematics import _bound_values, _fsym, _label_items, _nonces, _shuffled


def gen_uncertain_symbolic_reasoning(rng: random.Random, ctx):
    """Symbolic derivation that carries uncertainty exactly.

    Half the episodes ask for a *tightest bound* on a compound event given only
    the marginals — the Fréchet bounds, which are exact and which the four
    options deliberately confuse with one another, so no fixed heuristic (always
    the smaller marginal, always the sum) survives. The other half give
    independent events and ask which of four compound claims is most probable;
    probabilities are compared as exact rationals, never floats."""
    if rng.random() < 0.5:
        for _ in range(200):
            a = rng.randrange(15, 90, 5)
            b = rng.randrange(15, 90, 5)
            vals = _bound_values(a, b)
            if a == b or a + b == 100 or len(set(vals.values())) != 4:
                continue
            kind = rng.choice(list(vals))
            side, conn = kind.split("_")
            answer = vals[kind]
            options = _shuffled(rng, list(vals.values()))
            query = Pred("tightest_bound", Ident(side),
                         Pred(conn, Ident("A"), Ident("B")))
            obs = Rec(probabilities=Lst([Pred("percent", Ident("A"), Num(a)),
                                         Pred("percent", Ident("B"), Num(b))]),
                      options=Lst([Num(v) for v in options]),
                      query=query)
            return obs, options, answer, {"kind": "frechet_bound", "pa": a, "pb": b,
                                          "bound": kind, "value": answer}

    n_claims = ctx.at(4, 9, default=4)
    names = _nonces(rng, ctx.at(4, 7, default=4), 2)
    for _ in range(200):
        probs = {nm: rng.randrange(20, 90, 5) for nm in names}
        cands = []
        for _ in range(ctx.at(12, 40, default=12)):
            k = rng.randint(2, 3)
            sub = rng.sample(names, k)
            form = rng.choice(["and", "or", "and_not"])
            if form == "and":
                f = ("and", ("atom", sub[0]), ("atom", sub[1])) if k == 2 else \
                    ("and", ("atom", sub[0]), ("and", ("atom", sub[1]), ("atom", sub[2])))
                p = Fraction(1)
                for s in sub:
                    p *= Fraction(probs[s], 100)
            elif form == "or":
                f = ("or", ("atom", sub[0]), ("atom", sub[1])) if k == 2 else \
                    ("or", ("atom", sub[0]), ("or", ("atom", sub[1]), ("atom", sub[2])))
                q = Fraction(1)
                for s in sub:
                    q *= Fraction(100 - probs[s], 100)
                p = 1 - q
            else:
                f = ("and", ("atom", sub[0]), ("not", ("atom", sub[1])))
                p = Fraction(probs[sub[0]], 100) * Fraction(100 - probs[sub[1]], 100)
            if all(f != g for g, _ in cands):
                cands.append((f, p))
        if len(cands) < n_claims:
            continue
        picked = rng.sample(cands, n_claims)
        best = max(p for _, p in picked)
        if sum(1 for _, p in picked if p == best) != 1:
            continue
        order = [c for c in picked if c[1] == best] + [c for c in picked if c[1] != best]
        break
    else:                                                # pragma: no cover - construction
        probs = {nm: 50 for nm in names}
        order = [(("atom", nm), Fraction(1, 2)) for nm in names]

    shown, label_of = _label_items(rng, order)
    obs = Rec(probabilities=Lst([Pred("percent", Ident(nm), Num(probs[nm])) for nm in names]),
              independence=Pred("mutually_independent"),
              claims=Lst([Pred("claim", Ident(lab), _fsym(f)) for lab, (f, _) in shown]),
              query=Ident("most_probable_claim"))
    return obs, _shuffled(rng, [lab for lab, _ in shown]), label_of[0], {
        "kind": "most_probable", "probs": dict(probs),
        "best": str(order[0][1])}


class UncertainSymbolicReasoning(Lesson):
    """Exact bounds and comparisons over uncertain propositions."""

    id = "uncertain_symbolic_reasoning"
    level = 90
    tags = ("mathematics", "formal-reasoning")
    teaches = "exact bounds and comparisons over uncertain propositions"
    capabilities = ('probability', 'bounding', 'symbolic_arithmetic')
    axes = {'reasoning_depth': 4, 'compositional_depth': 3, 'ambiguity': 3}

    generate = staticmethod(gen_uncertain_symbolic_reasoning)
