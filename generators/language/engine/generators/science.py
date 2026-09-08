"""Support for the scientific-induction lessons.

Private. Ported alongside the lessons that use it; every function here is
called by at least one generator in :mod:`langcurriculum.lessons`.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Mapping, Sequence

from .._structure import Ident, Num, Pred, Term

NONCE_LETTERS = "kmtszlpvrn"


def _shuffled(rng: random.Random, xs: Sequence[Any]) -> list[Any]:
    """A shuffled copy. Every answer vocabulary in this module goes through it."""
    ys = list(xs)
    rng.shuffle(ys)
    return ys


def _nonce(rng: random.Random, n: int = 3) -> str:
    return "".join(rng.choice(NONCE_LETTERS) for _ in range(n))


def _labels(prefix: str, n: int) -> list[str]:
    """Labels are assigned *after* the candidates are shuffled, never before."""
    return [f"{prefix}{i + 1}" for i in range(n)]


_BIN: dict[str, Callable[[int, int], int]] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "mod": lambda a, b: a % b,
    "max": lambda a, b: max(a, b),
    "min": lambda a, b: min(a, b),
}


_UN: dict[str, Callable[[int], int]] = {
    "neg": lambda a: -a,
    "abs": lambda a: abs(a),
    "sq": lambda a: a * a,
}


def _eval(s: Term, env: Mapping[str, int]) -> int:
    """Evaluate a law symbol under a variable assignment."""
    if s.type == "num":
        return int(s.value)
    if s.type in ("ident", "token", "str"):
        return int(env[str(s.value)])
    head = str(s.value[0])
    args = [_eval(c, env) for c in s.children]
    if head in _UN and len(args) == 1:
        return _UN[head](args[0])
    if head in _BIN and len(args) == 2:
        return _BIN[head](args[0], args[1])
    raise ValueError(f"not an expression: {head}/{len(args)}")


def _add(a: Term, b: Term) -> Term:
    return Pred("add", a, b)


def _sub(a: Term, b: Term) -> Term:
    return Pred("sub", a, b)


def _mul(a: Term, b: Term) -> Term:
    return Pred("mul", a, b)


def _lin(coeff: int, var: str, const: int) -> Term:
    return _add(_mul(Num(coeff), Ident(var)), Num(const))


_CA_CELLS = 6


def _ca_step(state: tuple[int, ...], rule: int) -> tuple[int, ...]:
    """One synchronous update of an elementary cellular automaton on a ring."""
    n = len(state)
    return tuple((rule >> ((state[(i - 1) % n] << 2) | (state[i] << 1) | state[(i + 1) % n])) & 1
                 for i in range(n))


def _ca_fits(traj: Sequence[tuple[int, ...]], rule: int) -> bool:
    return all(_ca_step(traj[t], rule) == traj[t + 1] for t in range(len(traj) - 1))


def _random_law(rng: random.Random) -> tuple[Term, Callable[[int], int]]:
    """A law y = f(x) drawn from a small family, as symbol *and* callable."""
    kind = rng.choice(["linear", "quadratic", "kink", "periodic", "shifted"])
    a = rng.choice([-3, -2, -1, 1, 2, 3])
    b = rng.randint(-6, 6)
    c = rng.randint(-3, 3)
    x = Ident("x")
    if kind == "linear":
        return _add(_mul(Num(a), x), Num(b)), (lambda v: a * v + b)
    if kind == "quadratic":
        return _add(_mul(Num(a), Pred("sq", x)), Num(b)), (lambda v: a * v * v + b)
    if kind == "kink":
        return (_add(_mul(Num(a), Pred("abs", _sub(x, Num(c)))), Num(b)),
                (lambda v: a * abs(v - c) + b))
    if kind == "periodic":
        m = rng.choice([2, 3, 4])
        return (_add(_mul(Num(a), Pred("mod", x, Num(m))), Num(b)),
                (lambda v: a * (v % m) + b))
    return (_add(_mul(Num(a), _sub(x, Num(c))), _mul(Num(b), Num(1))),
            (lambda v: a * (v - c) + b))


def _random_core(rng: random.Random, vs: Sequence[str]) -> tuple[Term, Callable[[Mapping[str, int]], int]]:
    """The part of the law every hypothesis agrees on."""
    u, w = vs[0], vs[1]
    kind = rng.choice(["prod", "weighted", "diff", "square"])
    k = rng.choice([1, 2, 3])
    if kind == "prod":
        return _mul(Ident(u), Ident(w)), (lambda e: e[u] * e[w])
    if kind == "weighted":
        return (_add(_mul(Num(k), Ident(u)), Ident(w)), (lambda e: k * e[u] + e[w]))
    if kind == "diff":
        return _sub(Ident(u), _mul(Num(k), Ident(w))), (lambda e: e[u] - k * e[w])
    return _add(Pred("sq", Ident(u)), Ident(w)), (lambda e: e[u] * e[u] + e[w])


_CAUSES = ["noise", "hidden_variable", "boundary_condition"]


def _apply_perm(state: Sequence[int], perm: Sequence[int]) -> tuple[int, ...]:
    """``perm[i]`` is where the content of site ``i`` goes."""
    out = [0] * len(state)
    for i, v in enumerate(state):
        out[perm[i]] = v
    return tuple(out)


def _macro_eval(spec: Term, state: Sequence[int]) -> int:
    """Evaluate a candidate macrostate on a micro configuration."""
    head = str(spec.value[0])
    args = [int(c.value) for c in spec.children]
    if head == "sum_mod":
        return sum(state) % args[0]
    if head == "count_equal":
        return sum(1 for v in state if v == args[0])
    if head == "count_gt":
        return sum(1 for v in state if v > args[0])
    if head == "cell":
        return state[args[0]]
    if head == "max":
        return max(state)
    if head == "min":
        return min(state)
    if head == "range":
        return max(state) - min(state)
    if head == "half_sum_mod":
        return sum(state[: len(state) // 2]) % args[0]
    raise ValueError(f"unknown macrostate {head}")


def _closed(seq: Sequence[int]) -> bool:
    """Is this sequence a deterministic function of its own previous value?"""
    succ: dict[int, int] = {}
    for a, b in zip(seq, seq[1:]):
        if succ.setdefault(a, b) != b:
            return False
    return True


def _has_repeat(seq: Sequence[int]) -> bool:
    return len(set(seq[:-1])) < len(seq) - 1


def _transform(spec: Term, v: Sequence[int]) -> tuple[int, ...]:
    head = str(spec.value[0])
    args = [int(c.value) if c.type == "num" else c for c in spec.children]
    if head == "translate":
        return tuple(x + args[0] for x in v)
    if head == "scale":
        return tuple(x * args[0] for x in v)
    if head == "negate":
        return tuple(-x for x in v)
    if head == "reverse":
        return tuple(reversed(v))
    if head == "sort":
        return tuple(sorted(v))
    if head == "permute":
        p = [int(c.value) for c in spec.children[0].children]
        return tuple(v[p[i]] for i in range(len(v)))
    if head == "replace":
        out = list(v)
        out[args[0]] = args[1]
        return tuple(out)
    raise ValueError(f"unknown transformation {head}")


def _property(spec: Term, v: Sequence[int]) -> Any:
    head = str(spec.value[0])
    args = [int(c.value) for c in spec.children]
    if head == "sum":
        return sum(v)
    if head == "spread":
        return max(v) - min(v)
    if head == "multiset":
        return tuple(sorted(v))
    if head == "parity_of_sum":
        return sum(v) % 2
    if head == "at":
        return v[args[0]]
    if head == "count_positive":
        return sum(1 for x in v if x > 0)
    if head == "maximum":
        return max(v)
    raise ValueError(f"unknown property {head}")


_DIMS = ("mass", "length", "time")


def _dim_of(s: Term, table: Mapping[str, tuple[int, int, int]]) -> tuple[int, int, int] | None:
    """Dimension vector of an expression, or ``None`` if it has none."""
    if s.type == "num":
        return (0, 0, 0)
    if s.type in ("ident", "token", "str"):
        return table.get(str(s.value))
    head = str(s.value[0])
    if head == "pow":
        base = _dim_of(s.children[0], table)
        k = int(s.children[1].value)
        return None if base is None else tuple(k * e for e in base)     # type: ignore[return-value]
    parts = [_dim_of(c, table) for c in s.children]
    if any(p is None for p in parts):
        return None
    if head == "mul":
        return tuple(a + b for a, b in zip(*parts))                     # type: ignore[return-value]
    if head == "div":
        return tuple(a - b for a, b in zip(*parts))                     # type: ignore[return-value]
    if head == "add":
        return parts[0] if parts[0] == parts[1] else None
    raise ValueError(f"unknown dimensional form {head}")


def _equation_ok(eq: Term, table: Mapping[str, tuple[int, int, int]]) -> bool:
    lhs, rhs = eq.children
    dl, dr = _dim_of(lhs, table), _dim_of(rhs, table)
    return dl is not None and dr is not None and dl == dr
