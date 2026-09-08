"""``dimensional_analysis`` — algebra over symbolic dimensions.

Scientific induction and model discovery.
"""

from __future__ import annotations

import random

from .._structure import Ident, Lst, Num, Pred, Rec, Term
from ..lesson import Lesson
from ..generators.science import _DIMS, _dim_of, _equation_ok, _labels, _nonce, _shuffled


def gen_dimensional_analysis(rng: random.Random, ctx):
    """One of these four equations cannot be true whatever the constants are.

    Quantities carry symbolic dimensions given in a table; the candidate
    equations are built as products of powers, sometimes with a sum, and their
    left-hand quantities are *defined* to have whatever dimension makes them
    consistent — except for one, which is off by a perturbation, or which adds
    two terms of different dimension. Every name is a nonce, so no physics is
    recalled, only algebra over exponent vectors.
    """
    n_eq = ctx.at(4, 8, default=4)                 # equations to check, one of them impossible
    for _ in range(400):
        base = []
        while len(base) < 4:
            nm = _nonce(rng, 3)
            if nm not in base:
                base.append(nm)
        table: dict[str, tuple[int, int, int]] = {}
        for nm in base:
            d = tuple(rng.randint(-2, 2) for _ in range(3))
            if not any(d):
                d = (1, 0, 0)
            table[nm] = d                                                # type: ignore[assignment]
        ratio = _nonce(rng, 3)
        if ratio in table:
            continue
        table[ratio] = (0, 0, 0)                                         # a dimensionless one

        def monomial() -> Term:
            u, w = rng.sample(base, 2)
            e1, e2 = rng.choice([1, 2, -1]), rng.choice([1, 2, -1])
            left = Pred("pow", Ident(u), Num(e1))
            right = Pred("pow", Ident(w), Num(e2))
            return Pred("mul", left, right) if rng.random() < 0.6 else Pred("div", left, right)

        def twin(m: Term) -> Term:
            """A structurally different monomial with the *same* dimension:
            ``u^i · w^j`` and ``u^i / w^-j`` are the same quantity written twice."""
            head = str(m.value[0])
            a, b = m.children
            u, e1 = str(a.children[0].value), int(a.children[1].value)
            w, e2 = str(b.children[0].value), int(b.children[1].value)
            flip = "div" if head == "mul" else "mul"
            return Pred(flip, Pred("pow", Ident(u), Num(e1)), Pred("pow", Ident(w), Num(-e2)))

        bad_i = rng.randrange(n_eq)
        eqs, lhs_names = [], []
        ok = True
        for i in range(n_eq):
            rhs = monomial()
            if rng.random() < 0.45:                        # a sum, in good and bad alike
                rhs = Pred("add", rhs, Pred("mul", twin(rhs), Ident(ratio)))
            d = _dim_of(rhs, table)
            if d is None:
                ok = False
                break
            target = d
            if i == bad_i:                                 # the impossible one, by construction
                bump = [0, 0, 0]
                bump[rng.randrange(3)] = rng.choice([-2, -1, 1, 2])
                target = tuple(a + b for a, b in zip(target, bump))       # type: ignore[arg-type]
            nm = _nonce(rng, 3)
            if nm in table:
                ok = False
                break
            table[nm] = target                                            # type: ignore[assignment]
            lhs_names.append(nm)
            eqs.append(Pred("eq", Ident(nm), rhs))
        if not ok or len(eqs) != n_eq:
            continue
        verdicts = [_equation_ok(e, table) for e in eqs]
        if verdicts.count(False) != 1 or verdicts[bad_i]:
            continue
        order = _shuffled(rng, range(n_eq))
        labels = _labels("q", n_eq)
        answer = labels[order.index(bad_i)]
        names = _shuffled(rng, sorted(table))
        obs = Rec(dimensions=Lst([Pred("dim", Ident(nm), Num(table[nm][0]), Num(table[nm][1]),
                                       Num(table[nm][2])) for nm in names]),
                  base_dimensions=Lst([Ident(d) for d in _DIMS]),
                  equations=Lst([Pred("equation", Ident(labels[j]), eqs[i])
                                 for j, i in enumerate(order)]),
                  query=Ident("dimensionally_impossible"))
        hidden = {"answer": answer, "impossible": str(eqs[bad_i]),
                  "lhs_names": lhs_names}
        return obs, _shuffled(rng, labels), answer, hidden
    raise RuntimeError("dimensional_analysis: no admissible episode")


class DimensionalAnalysis(Lesson):
    """Algebra over symbolic dimensions."""

    id = "dimensional_analysis"
    level = 80
    tags = ("science", "induction", "model-discovery")
    teaches = "algebra over symbolic dimensions"
    capabilities = ('abstraction', 'scientific_induction', 'ontology_learning')
    axes = {'reasoning_depth': 4, 'compositional_depth': 4, 'lexical_novelty': 3}

    generate = staticmethod(gen_dimensional_analysis)
