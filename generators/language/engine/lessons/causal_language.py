"""``causal_language`` — causation vs correlation, decided by intervention.

Analogy, causality, planning, and programs.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec, Term
from ..lesson import Lesson
from ..generators.causal import _labels, _nonce_names, _options, _tree_values


def gen_causal_language(rng: random.Random, ctx):
    """Correlation is made *useless* here: the mechanisms are deterministic, so
    in the observational trace every pair of variables in a component is
    perfectly (anti-)correlated. Only the interventional blocks separate the
    claims, and they separate them exactly — under do(u=0)/do(u=1), a variable
    moves iff u is its causal ancestor."""
    deeper = ctx.at(0, 4, default=0)                          # extra links below C
    names = _nonce_names(rng, 7 + deeper)
    rng.shuffle(names)
    R, A, B, C, D, E, F = names[:7]
    chain = names[7:]
    parent = {A: R, B: R, C: A, D: B, F: E}
    below = C
    for v in chain:
        parent[v] = below
        below = v
    neg = {v: rng.randint(0, 1) for v in parent}
    order = [R, A, B, C, D, E, F] + chain
    main = [R, A, B, C, D] + chain

    ancestors: dict[str, list[str]] = {}
    for v in order:
        ancestors[v] = ([parent[v]] + ancestors[parent[v]]) if v in parent else []
    causal_pairs = [(u, v) for v in main for u in ancestors[v] if u in main]
    incomparable = [(x, y) for x in main for y in main
                    if x != y and x not in ancestors[y] and y not in ancestors[x]]

    true_claim = rng.choice(causal_pairs)
    pool = [(true_claim[1], true_claim[0])] + [p for p in incomparable if p != true_claim]
    rng.shuffle(pool)
    wrong = [pool[0]] + [p for p in pool[1:] if p != pool[0]][:2]
    if len(wrong) < 3:                                        # pragma: no cover - 5-node tree always has 8
        wrong = (wrong + incomparable)[:3]
    opts, correct = _options(rng, true_claim, wrong)
    labels = _labels("c", len(opts))

    # observational trace: all four exogenous configurations, in random order
    combos = [(a, b) for a in (0, 1) for b in (0, 1)]
    rng.shuffle(combos)
    trace: list[Term] = []
    for i, (r0, e0) in enumerate(combos):
        vals = _tree_values(parent, neg, {R: r0, E: e0}, order)
        for v in order:
            trace.append(Pred("obs", Num(i), Ident(v), Num(vals[v])))

    # interventional blocks: do(u=0) and do(u=1) for every candidate antecedent
    inter: list[Term] = []
    block = 0
    for u in sorted({o[0] for o in opts}):
        for w in (0, 1):
            seen: list[tuple[int, ...]] = []
            rows: list[dict[str, int]] = []
            for r0 in (0, 1):
                vals = _tree_values(parent, neg, {R: r0, E: 0}, order, forced={u: w})
                key = tuple(vals[v] for v in main)
                if key not in seen:
                    seen.append(key)
                    rows.append(vals)
            inter.append(Pred("do", Num(block), Ident(u), Num(w)))
            for j, vals in enumerate(rows):
                for v in main:
                    inter.append(Pred("after", Num(block), Num(j), Ident(v), Num(vals[v])))
            block += 1

    obs = Rec(variables=Lst([Ident(v) for v in order]),
              trace=Lst(trace),
              interventions=Lst(inter),
              claims=Lst([Pred("claim", Ident(lab), Ident(u), Ident(v))
                          for lab, (u, v) in zip(labels, opts)]),
              query=Pred("which_claim_is_causal"))
    hidden = {"edges": [[p, c] for c, p in sorted(parent.items())],
              "negations": {k: neg[k] for k in sorted(neg)},
              "true_claim": list(true_claim), "answer_label": labels[correct]}
    return obs, labels, labels[correct], hidden


class CausalLanguage(Lesson):
    """Causation vs correlation, decided by intervention."""

    id = "causal_language"
    level = 42
    tags = ("analogy", "causality", "planning", "programs")
    teaches = "causation vs correlation, decided by intervention"
    capabilities = ('causal_reasoning', 'scientific_induction', 'abstraction')
    axes = {'reasoning_depth': 4, 'world_complexity': 3, 'ambiguity': 2, 'compositional_depth': 2}
    answers = ['c0', 'c1', 'c2', 'c3']

    generate = staticmethod(gen_causal_language)
