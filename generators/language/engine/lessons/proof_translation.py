"""``proof_translation`` — the same derivation across two renamed calculi.

Mathematics and formal reasoning.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Pred, Rec
from ..lesson import Lesson
from ..generators.mathematics import _dag_isomorphisms, _nonces, _shuffled


def gen_proof_translation(rng: random.Random, ctx):
    """The same derivation, written twice in mutually unintelligible notation.

    Atoms, rule names and step labels are all renamed by unknown bijections and
    the target proof is listed in scrambled order, so nothing but the shape of
    the derivation connects the two. The episode is kept only when brute-force
    search finds a *single* structure-preserving correspondence, which then fixes
    the answer for any queried source step."""
    for _ in range(300):
        n = rng.randint(*ctx.span((4, 6), (6, 8)))
        prem: list[tuple[int, ...]] = []
        for i in range(n):
            if i < 2:
                prem.append(())
            elif rng.random() < 0.45 or i == 2:
                prem.append((rng.randrange(i),))
            else:
                prem.append(tuple(rng.sample(range(i), 2)))
        used = {n - 1}
        stack = [n - 1]
        while stack:
            i = stack.pop()
            for j in prem[i]:
                if j not in used:
                    used.add(j)
                    stack.append(j)
        if len(used) != n:
            continue
        src_rule_pool = _nonces(rng, 3, 2)
        rule_a = [src_rule_pool[0] if not prem[i] else rng.choice(src_rule_pool[1:]) for i in range(n)]
        sigma = _shuffled(rng, range(n))                 # source step i -> target step sigma[i]
        tgt_rule_pool = _nonces(rng, 3, 2, avoid=src_rule_pool)
        rmap = dict(zip(src_rule_pool, tgt_rule_pool))
        prem_b: list[tuple[int, ...]] = [()] * n
        rule_b: list[str] = [""] * n
        for i in range(n):
            prem_b[sigma[i]] = tuple(sigma[j] for j in prem[i])
            rule_b[sigma[i]] = rmap[rule_a[i]]
        isos = _dag_isomorphisms(prem, rule_a, prem_b, rule_b)
        if len(isos) == 1:
            break
    else:                                                # pragma: no cover - construction
        n = 4
        prem = [(), (), (0,), (1, 2)]
        src_rule_pool = _nonces(rng, 3, 2)
        rule_a = [src_rule_pool[0], src_rule_pool[0], src_rule_pool[1], src_rule_pool[2]]
        sigma = list(range(n))
        tgt_rule_pool = _nonces(rng, 3, 2, avoid=src_rule_pool)
        rmap = dict(zip(src_rule_pool, tgt_rule_pool))
        prem_b = [tuple(sigma[j] for j in prem[i]) for i in range(n)]
        rule_b = [rmap[r] for r in rule_a]

    atoms_a = _nonces(rng, n, 2, avoid=src_rule_pool + tgt_rule_pool)
    atoms_b = _nonces(rng, n, 2, avoid=src_rule_pool + tgt_rule_pool + atoms_a)
    sid = [f"s{i}" for i in range(n)]
    tid = [f"t{i}" for i in range(n)]

    src = [Pred("step", Ident(sid[i]), Ident(atoms_a[i]), Ident(rule_a[i]),
                Lst([Ident(sid[j]) for j in prem[i]])) for i in range(n)]
    tgt = [Pred("step", Ident(tid[k]), Ident(atoms_b[k]), Ident(rule_b[k]),
                Lst([Ident(tid[j]) for j in prem_b[k]])) for k in range(n)]
    k_query = rng.randrange(n)
    obs = Rec(source_proof=Lst(src), target_proof=Lst(_shuffled(rng, tgt)),
              query=Pred("corresponding_step", Ident(sid[k_query])))
    return obs, _shuffled(rng, tid), tid[sigma[k_query]], {
        "steps": n, "queried": sid[k_query], "correspondence": {sid[i]: tid[sigma[i]] for i in range(n)}}


class ProofTranslation(Lesson):
    """The same derivation across two renamed calculi."""

    id = "proof_translation"
    level = 86
    tags = ("mathematics", "formal-reasoning")
    teaches = "the same derivation across two renamed calculi"
    capabilities = ('structure_mapping', 'representation_independence')
    axes = {'compositional_depth': 4, 'reasoning_depth': 4, 'lexical_novelty': 4}

    generate = staticmethod(gen_proof_translation)
