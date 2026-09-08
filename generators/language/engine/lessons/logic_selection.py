"""``logic_selection`` — choosing the calculus a domain actually obeys.

Mathematics and formal reasoning.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.mathematics import _DISCRIMINATING, _REGIMES, _SCHEMAS, _SCHEMA_ATOMS, _VALIDITY, _fsym, _instantiate, _label_items, _matching_schemas, _nonces, _shuffled


def gen_logic_selection(rng: random.Random, ctx):
    """Which calculus does this domain run on?

    Four calculi are *defined in the episode* by the schemas they endorse and
    reject (their names are nonce, so nothing can be recalled). The domain is
    shown only through concrete inferences it licenses or blocks, written with
    unfamiliar atoms, so each observation has to be recognized as an instance of
    a schema before it can be matched against a calculus. Observations are kept
    only when exactly one calculus is consistent with all of them."""
    n_rules = ctx.at(5, 7, default=5)
    k_lo, k_hi = ctx.span((2, 4), (4, 7))
    for _ in range(300):
        names = rng.sample(_DISCRIMINATING, min(n_rules, len(_DISCRIMINATING)))
        profiles = {r: tuple(_VALIDITY[r][nm] for nm in names) for r in _REGIMES}
        if len(set(profiles.values())) != len(_REGIMES):
            continue
        true_regime = rng.choice(list(_REGIMES))
        k = rng.randint(k_lo, min(k_hi, len(names)))
        obs_names = rng.sample(names, k)
        consistent = [r for r in _REGIMES
                      if all(_VALIDITY[r][nm] == _VALIDITY[true_regime][nm] for nm in obs_names)]
        if consistent != [true_regime] and set(consistent) != {true_regime}:
            continue
        schema_by_name = {s[0]: s for s in _SCHEMAS}
        atoms = _nonces(rng, 2, 2)
        sub = dict(zip(_SCHEMA_ATOMS, atoms))
        instances = [(nm, _instantiate(schema_by_name[nm], sub)) for nm in obs_names]
        pool = [schema_by_name[nm] for nm in names]
        if any(_matching_schemas(p, c, pool, atoms) != [nm] for nm, (p, c) in instances):
            continue
        break
    else:                                                # pragma: no cover - construction
        names = _DISCRIMINATING[:5]
        true_regime = "classical"
        obs_names = names[:3]
        schema_by_name = {s[0]: s for s in _SCHEMAS}
        atoms = _nonces(rng, 2, 2)
        sub = dict(zip(_SCHEMA_ATOMS, atoms))
        instances = [(nm, _instantiate(schema_by_name[nm], sub)) for nm in obs_names]

    regimes = [true_regime] + [r for r in _REGIMES if r != true_regime]
    shown, label_of = _label_items(rng, regimes, prefix="L")
    rule_ids = {nm: f"r{i}" for i, nm in enumerate(_shuffled(rng, names))}
    schema_by_name = {s[0]: s for s in _SCHEMAS}

    rules_sym = Lst([Pred("schema", Ident(rule_ids[nm]),
                          Lst([_fsym(p) for p in schema_by_name[nm][1]]),
                          _fsym(schema_by_name[nm][2]))
                     for nm in sorted(names, key=lambda x: rule_ids[x])])
    calculi = Lst([Pred("calculus", Ident(lab),
                        Lst([Ident(rule_ids[nm]) for nm in names if _VALIDITY[r][nm]]),
                        Lst([Ident(rule_ids[nm]) for nm in names if not _VALIDITY[r][nm]]))
                   for lab, r in shown])
    domain = Lst(_shuffled(rng, [
        Pred("licenses" if _VALIDITY[true_regime][nm] else "blocks",
             Lst([_fsym(p) for p in prems]), _fsym(concl))
        for nm, (prems, concl) in instances]))
    obs = Rec(schemas=rules_sym, calculi=calculi, domain=domain,
              query=Ident("calculus_consistent_with_the_domain"))
    return obs, _shuffled(rng, [lab for lab, _ in shown]), label_of[0], {
        "regime": true_regime, "observed": list(obs_names), "rule_set": list(names)}


class LogicSelection(Lesson):
    """Choosing the calculus a domain actually obeys."""

    id = "logic_selection"
    level = 89
    tags = ("mathematics", "formal-reasoning")
    teaches = "choosing the calculus a domain actually obeys"
    capabilities = ('calculus_selection', 'schema_matching', 'validity')
    axes = {'reasoning_depth': 5, 'lexical_novelty': 4, 'ambiguity': 3}

    generate = staticmethod(gen_logic_selection)
