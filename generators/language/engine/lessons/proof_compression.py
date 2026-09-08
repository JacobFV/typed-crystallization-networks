"""``proof_compression`` — factoring a repeated subproof out of a long derivation.

Mathematics and formal reasoning.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec
from ..lesson import Lesson
from ..generators.mathematics import _derivation_costs, _horn_theory, _label_items, _ladder_theory, _proof_steps, _shuffled


def gen_proof_compression(rng: random.Random, ctx):
    """Two goals are proved by one long flat derivation with repeated subproofs.

    Factoring a shared subproof out as a named lemma pays its cost once instead
    of once per use; the answer is the proposition whose abstraction shortens the
    *total* derivation the most, with ties rejected. This is proof structure, not
    proof validity: every candidate is already true."""
    n_atoms = ctx.at(9, 14, default=9)            # a bigger theory means longer derivations
    for att in range(400):
        if att < 380:
            atoms, facts, rules = _horn_theory(rng, n_atoms=n_atoms, n_facts=3)
            d0 = _derivation_costs(facts, rules, atoms)
            tail = [a for a in atoms[5:] if d0[a] is not None]
            if len(tail) < 2:
                continue
            g1, g2 = rng.sample(tail, 2)
            cands_all = None
        else:                                            # pragma: no cover - budget
            atoms, facts, rules = _ladder_theory(rng, extra=True)
            d0 = _derivation_costs(facts, rules, atoms)
            g1, g2 = atoms[7], atoms[8]
            cands_all = atoms[3:7]
        total0 = d0[g1] + d0[g2]
        if total0 < 8:
            continue
        cands = cands_all or [a for a in atoms
                              if a not in facts and a not in (g1, g2) and d0[a] is not None]
        if len(cands) < 4:
            continue
        cands = list(cands) if cands_all else rng.sample(cands, 4)
        savings = []
        for c in cands:
            dc = _derivation_costs(list(facts) + [c], rules, atoms)
            new = d0[c] + (dc[g1] if dc[g1] is not None else d0[g1]) \
                        + (dc[g2] if dc[g2] is not None else d0[g2])
            savings.append(total0 - new)
        best = max(savings)
        if best <= 0 or savings.count(best) != 1:
            continue
        win = cands[savings.index(best)]
        order = [win] + [c for c in cands if c != win]
        steps = _proof_steps(g1, facts, rules, d0) + _proof_steps(g2, facts, rules, d0)
        break
    else:                                                # pragma: no cover - unreachable
        raise RuntimeError("proof_compression: no well-posed episode found")

    shown, label_of = _label_items(rng, order, prefix="k")
    obs = Rec(axioms=Lst([Pred("fact", Ident(a)) for a in _shuffled(rng, facts)]),
              rules=Lst([Pred("rule", Ident(nm), Ident(h), Lst([Ident(b) for b in body]))
                         for nm, h, body in _shuffled(rng, rules)]),
              goals=Lst([Ident(g1), Ident(g2)]),
              derivation=Lst([Pred("step", Num(i), Ident(a), Ident(nm),
                                   Lst([Ident(b) for b in body]))
                              for i, (nm, a, body) in enumerate(steps)]),
              candidates=Lst([Pred("abstraction", Ident(lab), Ident(a)) for lab, a in shown]),
              query=Ident("lemma_that_most_compresses_the_derivation"))
    return obs, _shuffled(rng, [a for _, a in shown]), order[0], {
        "goals": [g1, g2], "base_total": int(total0), "savings": [int(s) for s in savings],
        "derivation_length": len(steps)}


class ProofCompression(Lesson):
    """Factoring a repeated subproof out of a long derivation."""

    id = "proof_compression"
    level = 85
    tags = ("mathematics", "formal-reasoning")
    teaches = "factoring a repeated subproof out of a long derivation"
    capabilities = ('abstraction', 'proof_structure', 'cost_reasoning')
    axes = {'reasoning_depth': 5, 'compositional_depth': 4, 'discourse_horizon': 3}

    generate = staticmethod(gen_proof_compression)
