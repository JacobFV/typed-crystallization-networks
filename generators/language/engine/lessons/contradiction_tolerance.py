"""``contradiction_tolerance`` — what conflicting sources still support.

Mathematics and formal reasoning.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.mathematics import _closure, _label_items, _nonces, _shuffled


def gen_contradiction_tolerance(rng: random.Random, ctx):
    """Sources disagree; reason on anyway, without exploding.

    An atom reported both ways is *contested* and supports nothing. A claim is
    supported when it follows from the uncontested reports alone. Exactly one of
    the four candidates is supported; the distractors are a claim derivable only
    through a contested atom, a contested atom itself, and a claim that never
    follows at all — so the lesson separates paraconsistent reasoning from both
    explosion and blanket scepticism."""
    n_atoms = ctx.at(8, 14, default=8)
    for _ in range(300):
        atoms = _nonces(rng, n_atoms, 2)
        srcs = _nonces(rng, 3, 2, avoid=atoms)
        rules: list[tuple[str, str, tuple[str, ...]]] = []
        rnames = _nonces(rng, n_atoms, 2, avoid=atoms + srcs)
        k = 0
        for i in range(3, n_atoms):
            for _ in range(rng.randint(1, 2)):
                if k >= len(rnames):
                    break
                rules.append((rnames[k], atoms[i], tuple(rng.sample(atoms[:i], rng.randint(1, 2)))))
                k += 1
        pos = rng.sample(atoms[:4], 3)
        contested = rng.choice(pos)
        reports = [(rng.choice(srcs), a, True) for a in pos]
        reports.append((rng.choice([s for s in srcs if s != reports[pos.index(contested)][0]]),
                        contested, False))
        safe_base = [a for a in pos if a != contested]
        all_base = list(pos)
        safe = _closure(safe_base, rules)
        loose = _closure(all_base, rules)
        # Filtered from `atoms`, which is a list, and not from the closures,
        # which are sets. Iterating a set of strings orders them by hash, and
        # Python salts string hashing per process -- so `rng.choice` over the
        # result picked a different atom in every run and the same seed gave a
        # different episode each time the file was regenerated.
        supported = [a for a in atoms if a in safe and a != contested]
        only_loose = [a for a in atoms if a in loose and a not in safe]
        never = [a for a in atoms if a not in loose]
        if not supported or not only_loose or not never:
            continue
        cands = [rng.choice(supported), rng.choice(only_loose), contested, rng.choice(never)]
        if len(set(cands)) != 4:
            continue
        break
    else:                                                # pragma: no cover - construction
        atoms = _nonces(rng, 8, 2)
        srcs = _nonces(rng, 3, 2, avoid=atoms)
        rules, reports = [], [(srcs[0], atoms[0], True)]
        cands, contested = atoms[:4], atoms[0]

    shown, _ = _label_items(rng, cands, prefix="q")
    obs = Rec(reports=Lst(_shuffled(rng, [
                  Pred("says", Ident(s), Ident(a) if pol else Pred("not", Ident(a)))
                  for s, a, pol in reports])),
              rules=Lst([Pred("rule", Ident(nm), Ident(h), Lst([Ident(b) for b in body]))
                         for nm, h, body in _shuffled(rng, rules)]),
              candidates=Lst([Pred("candidate", Ident(lab), Ident(a)) for lab, a in shown]),
              query=Ident("claim_still_supported_without_explosion"))
    return obs, _shuffled(rng, cands), cands[0], {
        "contested": contested, "n_rules": len(rules)}


class ContradictionTolerance(Lesson):
    """What conflicting sources still support."""

    id = "contradiction_tolerance"
    level = 93
    tags = ("mathematics", "formal-reasoning")
    teaches = "what conflicting sources still support"
    capabilities = ('paraconsistency', 'provenance', 'forward_chaining')
    axes = {'reasoning_depth': 4, 'ambiguity': 4, 'world_complexity': 3}

    generate = staticmethod(gen_contradiction_tolerance)
