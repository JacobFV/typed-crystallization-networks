"""Support for the mathematics and formal-reasoning lessons.

Private. Ported alongside the lessons that use it; every function here is
called by at least one generator in :mod:`langcurriculum.lessons`.
"""

from __future__ import annotations

import random
from itertools import permutations
from itertools import product
from typing import Any, Callable, Mapping, Sequence

from .._structure import Ident, Lst, Num, Pred, Term

_CONS = "bdgkmnprstvz"


_VOWS = "aeiou"


def _shuffled(rng: random.Random, xs) -> list:
    ys = list(xs)
    rng.shuffle(ys)
    return ys


def _nonce(rng: random.Random, syllables: int = 2) -> str:
    return "".join(rng.choice(_CONS) + rng.choice(_VOWS) for _ in range(syllables))


def _nonces(rng: random.Random, k: int, syllables: int = 2, avoid: Sequence[str] = ()) -> list[str]:
    out: list[str] = []
    seen = set(avoid)
    for _ in range(400):
        if len(out) == k:
            break
        w = _nonce(rng, syllables)
        if w not in seen:
            seen.add(w)
            out.append(w)
    while len(out) < k:                                  # pragma: no cover - exhaustion
        out.append(f"x{len(out)}")
    return out


def _label_items(rng: random.Random, items: Sequence[Any], prefix: str = "c"):
    """Attach labels to ``items`` in a random order.

    ``items[0]`` is by convention the correct one; the returned ``answer`` label
    is therefore uniform over the labels, which is what keeps a constant guesser
    at chance no matter how the generator ordered its candidates.
    """
    n = len(items)
    order = _shuffled(rng, range(n))                     # slot j shows items[order[j]]
    labels = [f"{prefix}{j}" for j in range(n)]
    shown = [(labels[j], items[order[j]]) for j in range(n)]
    label_of = {order[j]: labels[j] for j in range(n)}
    return shown, label_of


def _fsym(f) -> Term:
    k = f[0]
    if k == "atom":
        return Ident(f[1])
    if k == "not":
        return Pred("not", _fsym(f[1]))
    return Pred(k, _fsym(f[1]), _fsym(f[2]))


def _atoms_of(f) -> set[str]:
    if f[0] == "atom":
        return {f[1]}
    if f[0] == "not":
        return _atoms_of(f[1])
    return _atoms_of(f[1]) | _atoms_of(f[2])


def _cval(f, v: Mapping[str, bool]) -> bool:
    k = f[0]
    if k == "atom":
        return bool(v[f[1]])
    if k == "not":
        return not _cval(f[1], v)
    if k == "and":
        return _cval(f[1], v) and _cval(f[2], v)
    if k == "or":
        return _cval(f[1], v) or _cval(f[2], v)
    return (not _cval(f[1], v)) or _cval(f[2], v)


def _assignments(atoms: Sequence[str]):
    for bits in product((False, True), repeat=len(atoms)):
        yield dict(zip(atoms, bits))


def _sat(formulas: Sequence[Any], atoms: Sequence[str]) -> bool:
    return any(all(_cval(f, v) for f in formulas) for v in _assignments(atoms))


def _rand_formula(rng: random.Random, atoms: Sequence[str], depth: int):
    if depth <= 0 or rng.random() < 0.25:
        a = ("atom", rng.choice(atoms))
        return ("not", a) if rng.random() < 0.35 else a
    op = rng.choice(["and", "or", "imp", "not"])
    if op == "not":
        return ("not", _rand_formula(rng, atoms, depth - 1))
    return (op, _rand_formula(rng, atoms, depth - 1), _rand_formula(rng, atoms, depth - 1))


def _cond_test(cond) -> Callable[[Sequence[int]], bool]:
    kind = cond[0]
    if kind == "has_even":
        return lambda s: any(x % 2 == 0 for x in s)
    if kind == "has_odd":
        return lambda s: any(x % 2 == 1 for x in s)
    if kind == "sum_even":
        return lambda s: sum(s) % 2 == 0
    if kind == "size_at_least":
        return lambda s, k=cond[1]: len(s) >= k
    if kind == "max_at_least":
        return lambda s, k=cond[1]: max(s) >= k
    if kind == "min_at_most":
        return lambda s, k=cond[1]: min(s) <= k
    if kind == "contains":
        return lambda s, v=cond[1]: v in s
    raise ValueError(kind)


def _cond_sym(cond) -> Term:
    return Pred(cond[0], *(Num(x) for x in cond[1:]))


def _rand_cond(rng: random.Random):
    kind = rng.choice(["has_even", "has_odd", "sum_even", "size_at_least",
                       "max_at_least", "min_at_most", "contains"])
    if kind == "size_at_least":
        return (kind, rng.randint(2, 4))
    if kind == "max_at_least":
        return (kind, rng.randint(4, 8))
    if kind == "min_at_most":
        return (kind, rng.randint(2, 5))
    if kind == "contains":
        return (kind, rng.randint(1, 9))
    return (kind,)


def _entails(premises: Sequence[Any], claim, atoms: Sequence[str]) -> bool:
    return all(_cval(claim, v) for v in _assignments(atoms) if all(_cval(p, v) for p in premises))


def _rewrite(s: str, lhs: str, rhs: str) -> str | None:
    i = s.find(lhs)
    return None if i < 0 else s[:i] + rhs + s[i + len(lhs):]


def _reach(start: str, rules: Sequence[tuple[str, str, str]], max_len: int, max_depth: int) -> dict[str, int]:
    dist = {start: 0}
    frontier = [start]
    for d in range(1, max_depth + 1):
        nxt = []
        for s in frontier:
            for _, lhs, rhs in rules:
                t = _rewrite(s, lhs, rhs)
                if t is None or len(t) > max_len or t in dist:
                    continue
                dist[t] = d
                nxt.append(t)
        frontier = nxt
        if not frontier:
            break
    return dist


_FALLBACK_RULES = [("a", "c"), ("b", "d"), ("c", "e"), ("ab", "f")]


def _derivation_costs(facts: Sequence[str], rules: Sequence[tuple[str, str, tuple[str, ...]]],
                      order: Sequence[str]) -> dict[str, int | None]:
    """Minimal *proof-tree size* for every atom: the number of rule applications
    in the smallest derivation, counting a shared subproof once per use.

    Because the atom order is a topological order of the rule graph, one sweep
    computes the fixpoint exactly."""
    d: dict[str, int | None] = {}
    fset = set(facts)
    for a in order:
        if a in fset:
            d[a] = 0
            continue
        best: int | None = None
        for _, head, body in rules:
            if head != a or any(d.get(b) is None for b in body):
                continue
            c = 1 + sum(d[b] for b in body)              # type: ignore[misc]
            if best is None or c < best:
                best = c
        d[a] = best
    return d


def _proof_steps(a: str, facts: Sequence[str], rules, d) -> list[tuple[str, str, tuple[str, ...]]]:
    """Expand the minimal derivation of ``a`` into a linear list of steps."""
    if a in set(facts):
        return [("given", a, ())]
    best, choice = None, None
    for nm, head, body in rules:
        if head != a or any(d.get(b) is None for b in body):
            continue
        c = 1 + sum(d[b] for b in body)
        if best is None or c < best:
            best, choice = c, (nm, head, body)
    if choice is None:                                   # pragma: no cover - guarded by caller
        return []
    steps: list[tuple[str, str, tuple[str, ...]]] = []
    for b in choice[2]:
        steps += _proof_steps(b, facts, rules, d)
    steps.append(choice)
    return steps


def _horn_theory(rng: random.Random, n_atoms: int = 8, n_facts: int = 3, window: int = 3):
    """A random acyclic Horn theory whose rule bodies are drawn from *recent*
    atoms, so derivations compose into deep trees with shared subproofs rather
    than collapsing onto the axioms in one step."""
    atoms = _nonces(rng, n_atoms, 2)
    facts = atoms[:n_facts]
    rules: list[tuple[str, str, tuple[str, ...]]] = []
    names = _nonces(rng, 2 * (n_atoms - n_facts), 2, avoid=atoms)
    k = 0
    for i in range(n_facts, n_atoms):
        pool = atoms[max(0, i - window):i]
        if len(pool) < 2:
            pool = atoms[:i]
        for _ in range(rng.randint(1, 2)):
            size = 2 if len(pool) >= 2 and rng.random() < 0.75 else 1
            rules.append((names[k], atoms[i], tuple(rng.sample(pool, size))))
            k += 1
    return atoms, facts, rules


def _ladder_theory(rng: random.Random, extra: bool = False):
    """A fixed theory with heavily shared subproofs, used when the random search
    below runs out of budget: its lemma savings are strictly ordered, so the
    episode it produces is as well-posed as a sampled one."""
    n = 9 if extra else 8
    names = _nonces(rng, 6, 2)
    atoms = _nonces(rng, n, 2, avoid=names)
    f0, f1, f2 = atoms[:3]
    u, v, w, z, g = atoms[3:8]
    rules = [(names[0], u, (f0, f1)), (names[1], v, (u, f2)), (names[2], w, (u, v)),
             (names[3], z, (v, w)), (names[4], g, (w, z))]
    if extra:
        rules.append((names[5], atoms[8], (v, f0)))
    return atoms, atoms[:3], rules


def _dag_isomorphisms(prem_a: Sequence[tuple[int, ...]], rule_a: Sequence[str],
                      prem_b: Sequence[tuple[int, ...]], rule_b: Sequence[str]) -> list[tuple[int, ...]]:
    """Every step-correspondence preserving premise structure and rule identity.

    Rule *names* differ between calculi, so only the partition they induce is
    required to match: the correspondence must map same-rule steps to same-rule
    steps, bijectively."""
    n = len(prem_a)
    out = []
    for pi in permutations(range(n)):
        ok = True
        for i in range(n):
            if tuple(pi[j] for j in prem_a[i]) != prem_b[pi[i]]:
                ok = False
                break
        if not ok:
            continue
        fwd: dict[str, str] = {}
        back: dict[str, str] = {}
        for i in range(n):
            r, s = rule_a[i], rule_b[pi[i]]
            if fwd.setdefault(r, s) != s or back.setdefault(s, r) != r:
                ok = False
                break
        if ok:
            out.append(pi)
    return out


_CLAIMS = [
    ("implication", lambda dom, P, Q, R: all((not P[x]) or Q[x] for x in dom)),
    ("converse", lambda dom, P, Q, R: all((not Q[x]) or P[x] for x in dom)),
    ("symmetry", lambda dom, P, Q, R: all((not R[(x, y)]) or R[(y, x)] for x in dom for y in dom)),
    ("transitivity", lambda dom, P, Q, R: all((not (R[(x, y)] and R[(y, z)])) or R[(x, z)]
                                              for x in dom for y in dom for z in dom)),
    ("irreflexivity", lambda dom, P, Q, R: all(not R[(x, x)] for x in dom)),
    ("seriality", lambda dom, P, Q, R: all(any(R[(x, y)] for y in dom) for x in dom)),
    ("relation_implies_p", lambda dom, P, Q, R: all((not R[(x, y)]) or P[x] for x in dom for y in dom)),
]


_CLAIM_SYM = {
    "implication": Pred("forall", Ident("x"), Pred("imp", Pred("P", Ident("x")), Pred("Q", Ident("x")))),
    "converse": Pred("forall", Ident("x"), Pred("imp", Pred("Q", Ident("x")), Pred("P", Ident("x")))),
    "symmetry": Pred("forall", Ident("x"), Ident("y"),
                     Pred("imp", Pred("R", Ident("x"), Ident("y")), Pred("R", Ident("y"), Ident("x")))),
    "transitivity": Pred("forall", Ident("x"), Ident("y"), Ident("z"),
                         Pred("imp", Pred("and", Pred("R", Ident("x"), Ident("y")),
                                          Pred("R", Ident("y"), Ident("z"))),
                              Pred("R", Ident("x"), Ident("z")))),
    "irreflexivity": Pred("forall", Ident("x"), Pred("not", Pred("R", Ident("x"), Ident("x")))),
    "seriality": Pred("forall", Ident("x"), Pred("exists", Ident("y"), Pred("R", Ident("x"), Ident("y")))),
    "relation_implies_p": Pred("forall", Ident("x"), Ident("y"),
                               Pred("imp", Pred("R", Ident("x"), Ident("y")), Pred("P", Ident("x")))),
}


def _rand_model(rng: random.Random, dom: Sequence[str]):
    P = {x: rng.random() < 0.5 for x in dom}
    Q = {x: rng.random() < 0.5 for x in dom}
    R = {(x, y): rng.random() < 0.4 for x in dom for y in dom}
    return P, Q, R


def _model_sym(label: str, dom: Sequence[str], P, Q, R) -> Term:
    facts = [Pred("P", Ident(x)) for x in dom if P[x]]
    facts += [Pred("Q", Ident(x)) for x in dom if Q[x]]
    facts += [Pred("R", Ident(x), Ident(y)) for x in dom for y in dom if R[(x, y)]]
    return Pred("model", Ident(label), Lst([Pred("domain", *[Ident(x) for x in dom])] + facts))


_REGIMES: dict[str, dict[str, Any]] = {
    "classical": {"values": (0, 2), "designated": (2,),
                  "neg": lambda a: 2 - a, "imp": lambda a, b: max(2 - a, b)},
    "strong_kleene": {"values": (0, 1, 2), "designated": (2,),
                      "neg": lambda a: 2 - a, "imp": lambda a, b: max(2 - a, b)},
    "heyting": {"values": (0, 1, 2), "designated": (2,),
                "neg": lambda a: 2 if a == 0 else 0,
                "imp": lambda a, b: 2 if a <= b else b},
    "paraconsistent": {"values": (0, 1, 2), "designated": (1, 2),
                       "neg": lambda a: 2 - a, "imp": lambda a, b: max(2 - a, b)},
}


def _mv_eval(f, v: Mapping[str, int], reg: Mapping[str, Any]) -> int:
    k = f[0]
    if k == "atom":
        return v[f[1]]
    if k == "not":
        return reg["neg"](_mv_eval(f[1], v, reg))
    a = _mv_eval(f[1], v, reg)
    b = _mv_eval(f[2], v, reg)
    if k == "and":
        return min(a, b)
    if k == "or":
        return max(a, b)
    return reg["imp"](a, b)


_A = ("atom", "A")


_B = ("atom", "B")


def _n(f):
    return ("not", f)


_SCHEMAS: list[tuple[str, tuple, Any]] = [
    ("excluded_middle", (), ("or", _A, _n(_A))),
    ("double_negation_elim", (_n(_n(_A)),), _A),
    ("double_negation_intro", (_A,), _n(_n(_A))),
    ("ex_falso", (_A, _n(_A)), _B),
    ("disjunctive_syllogism", (("or", _A, _B), _n(_A)), _B),
    ("modus_ponens", (_A, ("imp", _A, _B)), _B),
    ("modus_tollens", (("imp", _A, _B), _n(_B)), _n(_A)),
    ("contraposition", (("imp", _A, _B),), ("imp", _n(_B), _n(_A))),
    ("noncontradiction", (), _n(("and", _A, _n(_A)))),
    ("conditional_proof", (_B,), ("imp", _A, _B)),
    ("addition", (_A,), ("or", _A, _B)),
    ("simplification", (("and", _A, _B),), _A),
    ("affirming_consequent", (("imp", _A, _B), _B), _A),
    ("disjunct_extraction", (("or", _A, _B),), _A),
    ("weakening", (_A,), _B),
    ("conjunction_from_disjunction", (("or", _A, _B),), ("and", _A, _B)),
    ("implication_reversal", (("imp", _A, _B),), ("imp", _B, _A)),
]


_SCHEMA_ATOMS = ("A", "B")


def _schema_valid(schema, regime: str) -> bool:
    reg = _REGIMES[regime]
    des = set(reg["designated"])
    _, prems, concl = schema
    for combo in product(reg["values"], repeat=len(_SCHEMA_ATOMS)):
        v = dict(zip(_SCHEMA_ATOMS, combo))
        if all(_mv_eval(p, v, reg) in des for p in prems) and _mv_eval(concl, v, reg) not in des:
            return False
    return True


_VALIDITY: dict[str, dict[str, bool]] = {
    r: {s[0]: _schema_valid(s, r) for s in _SCHEMAS} for r in _REGIMES
}


_DISCRIMINATING = [s[0] for s in _SCHEMAS
                   if len({_VALIDITY[r][s[0]] for r in _REGIMES}) > 1]


def _formula_pool() -> list[Any]:
    seen: list[Any] = []
    for _, prems, concl in _SCHEMAS:
        for f in list(prems) + [concl]:
            if f not in seen:
                seen.append(f)
    return seen


_POOL = _formula_pool()


def _instantiate(schema, sub: Mapping[str, str]):
    def go(f):
        if f[0] == "atom":
            return ("atom", sub[f[1]])
        if f[0] == "not":
            return ("not", go(f[1]))
        return (f[0], go(f[1]), go(f[2]))
    _, prems, concl = schema
    return tuple(go(p) for p in prems), go(concl)


def _matching_schemas(prems, concl, schemas, atoms: Sequence[str]) -> list[str]:
    """Which schemas the given concrete inference is an instance of, under *some*
    renaming of its atoms — the renaming is never disclosed, so recognizing the
    schema is structural work."""
    out = []
    for s in schemas:
        for perm in permutations(range(len(_SCHEMA_ATOMS))):
            sub = {_SCHEMA_ATOMS[i]: atoms[perm[i]] for i in range(len(_SCHEMA_ATOMS))}
            ip, ic = _instantiate(s, sub)
            if ip == tuple(prems) and ic == concl:
                out.append(s[0])
                break
    return out


def _bound_values(a: int, b: int) -> dict[str, int]:
    return {"lower_and": max(0, a + b - 100), "upper_and": min(a, b),
            "lower_or": max(a, b), "upper_or": min(100, a + b)}


def _closure(base: Sequence[str], rules: Sequence[tuple[str, str, tuple[str, ...]]]) -> set[str]:
    out = set(base)
    changed = True
    while changed:
        changed = False
        for _, head, body in rules:
            if head not in out and all(b in out for b in body):
                out.add(head)
                changed = True
    return out
