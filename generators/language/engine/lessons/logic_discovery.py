"""``logic_discovery`` — inducing which inference rules a world licenses.

Mathematics and formal reasoning.
"""

from __future__ import annotations

import random
from itertools import product

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.mathematics import _DISCRIMINATING, _POOL, _REGIMES, _SCHEMAS, _SCHEMA_ATOMS, _fsym, _label_items, _mv_eval, _shuffled


def gen_logic_discovery(rng: random.Random, ctx):
    """Induce a world's logic from its epistemic dynamics, then apply it.

    The observation lists every distinguishable *situation* of the world and
    exactly which formulas that situation accepts — a complete, finite record of
    the world's acceptance behaviour. Validity is therefore decidable by reading
    the table: a rule is valid iff no listed situation accepts all its premises
    while rejecting its conclusion. The generated regime is one of classical,
    strong-Kleene, Heyting or paraconsistent, so the same famous schema (double
    negation elimination, excluded middle, ex falso, disjunctive syllogism) is
    valid in one episode and invalid in the next: prior classical reflexes are
    actively punished."""
    n_wrong = ctx.at(3, 8, default=3)             # invalid rules to tell the valid one from
    for _ in range(200):
        regime = rng.choice(list(_REGIMES))
        reg = _REGIMES[regime]
        des = set(reg["designated"])
        sits: list[frozenset[int]] = []
        for combo in product(reg["values"], repeat=len(_SCHEMA_ATOMS)):
            v = dict(zip(_SCHEMA_ATOMS, combo))
            acc = frozenset(i for i, f in enumerate(_POOL) if _mv_eval(f, v, reg) in des)
            if acc not in sits:
                sits.append(acc)
        idx = {f: i for i, f in enumerate(_POOL)}

        def valid(schema, sits=sits, idx=idx) -> bool:
            _, prems, concl = schema
            ps = [idx[p] for p in prems]
            c = idx[concl]
            return all(c in s for s in sits if all(p in s for p in ps))

        ok = [s for s in _SCHEMAS if valid(s)]
        bad = [s for s in _SCHEMAS if not valid(s)]
        if not ok or len(bad) < n_wrong:
            continue
        ok_d = [s for s in ok if s[0] in _DISCRIMINATING] or ok
        bad_d = [s for s in bad if s[0] in _DISCRIMINATING]
        answer_s = rng.choice(ok_d)
        others = rng.sample(bad_d, min(2, len(bad_d)))
        rest = [s for s in bad if s not in others]
        others += rng.sample(rest, n_wrong - len(others))
        break
    else:                                                # pragma: no cover - construction
        regime = "classical"
        sits = [frozenset()]
        answer_s, others = _SCHEMAS[10], _SCHEMAS[12:15]

    cands = [answer_s] + others
    shown, label_of = _label_items(rng, cands)
    obs = Rec(formulas=Lst([Pred("formula", Ident(f"f{i}"), _fsym(f)) for i, f in enumerate(_POOL)]),
              situations=Lst([Pred("situation", Ident(f"w{i}"),
                                   Lst([Ident(f"f{j}") for j in sorted(acc)]))
                              for i, acc in enumerate(sits)]),
              candidate_rules=Lst([
                  Pred("rule", Ident(lab),
                       Lst([Ident(f"f{_POOL.index(p)}") for p in s[1]]),
                       Ident(f"f{_POOL.index(s[2])}"))
                  for lab, s in shown]),
              query=Ident("rule_valid_in_every_situation"))
    return obs, _shuffled(rng, [lab for lab, _ in shown]), label_of[0], {
        "regime": regime, "answer_schema": answer_s[0],
        "distractors": [s[0] for s in others], "n_situations": len(sits)}


class LogicDiscovery(Lesson):
    """Inducing which inference rules a world licenses."""

    id = "logic_discovery"
    level = 88
    tags = ("mathematics", "formal-reasoning")
    teaches = "inducing which inference rules a world licenses"
    capabilities = ('rule_induction', 'semantics_induction', 'validity')
    axes = {'reasoning_depth': 5, 'ambiguity': 3, 'world_complexity': 4}

    generate = staticmethod(gen_logic_discovery)
