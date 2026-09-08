"""Support for the reflective-computation, protocol and narrative lessons.

Private. Ported alongside the lessons that use it; every function here is
called by at least one generator in :mod:`langcurriculum.lessons`.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..binding import SlotVocabulary


NONCE_LETTERS = "kmtszlpvrgd"


_SIZES = SlotVocabulary("size", ["small", "big", "huge"])


def _shuffled(rng: random.Random, xs: Iterable[Any]) -> list[Any]:
    ys = list(xs)
    rng.shuffle(ys)
    return ys


def _nonces(rng: random.Random, k: int, n: int = 3) -> list[str]:
    """``k`` distinct nonce words — a fresh vocabulary every episode."""
    out: list[str] = []
    while len(out) < k:
        w = "".join(rng.choice(NONCE_LETTERS) for _ in range(n))
        if w not in out:
            out.append(w)
    return out


def _num_options(rng: random.Random, correct: int, near: Sequence[int], k: int = 4) -> list[int]:
    """A shuffled numeric vocabulary containing ``correct`` and ``k-1`` others.

    ``near`` holds the *interesting* wrong answers (the value a shallower
    procedure would produce); offsets fill in only if those collide.
    """
    opts = [correct]
    for v in near:
        if len(opts) >= k:
            break
        if v not in opts:
            opts.append(v)
    d = 1
    while len(opts) < k:
        for v in (correct + d, correct - d):
            if v not in opts and len(opts) < k:
                opts.append(v)
        d += 1
    return _shuffled(rng, opts)


def _labels(rng: random.Random, prefix: str, k: int) -> list[str]:
    """``k`` option labels in random order, so a label never predicts anything."""
    return _shuffled(rng, [f"{prefix}{i}" for i in range(1, k + 1)])


_ARITH = ("add", "sub", "mul")


def _arith(kind: str, x: int, a: int) -> int:
    return x + a if kind == "add" else x - a if kind == "sub" else x * a


def _run_prog(prog: Sequence[str], sem: Mapping[str, tuple[str, int]], x: int) -> int:
    for t in prog:
        kind, a = sem[t]
        x = _arith(kind, x, a)
    return x


def _rewrite(rules: Sequence[tuple[str, str]], seq: Sequence[str]) -> list[str]:
    """First matching rule wins; unmatched tokens pass through unchanged."""
    m: dict[str, str] = {}
    for a, b in rules:
        m.setdefault(a, b)
    return [m.get(t, t) for t in seq]


def _cyk_count(w: Sequence[str], lex: Sequence[tuple[str, str]],
               bins: Sequence[tuple[str, str, str]], start: str = "S") -> int:
    """Number of distinct derivation trees of ``w`` from ``start``."""
    n = len(w)
    if n == 0:
        return 0
    cnt: list[list[dict[str, int]]] = [[{} for _ in range(n + 1)] for _ in range(n + 1)]
    for i, ch in enumerate(w):
        d = cnt[i][i + 1]
        for a, t in lex:
            if t == ch:
                d[a] = d.get(a, 0) + 1
    for span in range(2, n + 1):
        for i in range(0, n - span + 1):
            j = i + span
            d = cnt[i][j]
            for k in range(i + 1, j):
                for a, b, c in bins:
                    lb = cnt[i][k].get(b, 0)
                    if not lb:
                        continue
                    rc = cnt[k][j].get(c, 0)
                    if rc:
                        d[a] = d.get(a, 0) + lb * rc
    return cnt[0][n].get(start, 0)


def _coin_cost(target: int, coins: Sequence[int]) -> int:
    """Fewest words whose values sum to ``target`` (1 is always in ``coins``)."""
    inf = 10 ** 6
    best = [0] + [inf] * target
    for n in range(1, target + 1):
        for c in coins:
            if c <= n and best[n - c] + 1 < best[n]:
                best[n] = best[n - c] + 1
    return best[target]


def _dsl_cost(seq: Sequence[str], macros: Sequence[Sequence[str]]) -> int:
    """Fewest tokens for ``seq`` when any macro may stand for its expansion."""
    n = len(seq)
    inf = 10 ** 6
    best = [0] + [inf] * n
    for i in range(1, n + 1):
        best[i] = best[i - 1] + 1
        for m in macros:
            k = len(m)
            if i >= k and list(seq[i - k:i]) == list(m) and best[i - k] + 1 < best[i]:
                best[i] = best[i - k] + 1
    return best[n]


def _expand(items: Sequence[tuple], macros: Mapping[str, Sequence[tuple]], depth: int = 0) -> list[str]:
    """Compile: macro calls and ``times`` counts expand to a primitive stream."""
    out: list[str] = []
    if depth > 6:
        return out
    for it in items:
        if it[0] == "rep":
            out += _expand([("tok", it[2])], macros, depth + 1) * it[1]
        elif it[1] in macros:
            out += _expand(macros[it[1]], macros, depth + 1)
        else:
            out.append(it[1])
    return out


_SEM_KINDS = ("add", "sub", "mul", "max", "min", "set")


def _sem_apply(kind: str, x: int, a: int) -> int:
    if kind == "add":
        return x + a
    if kind == "sub":
        return x - a
    if kind == "mul":
        return x * a
    if kind == "max":
        return max(x, a)
    if kind == "min":
        return min(x, a)
    return a


def _run_ops(prog: Sequence[tuple[str, int]], assign: Mapping[str, str], x: int) -> int:
    for op, a in prog:
        x = _sem_apply(assign[op], x, a)
    return x


def _pure_equilibria(n: int, payoff: Callable[[tuple[int, ...], int], int]) -> list[tuple[int, ...]]:
    """Every pure-strategy Nash profile of a binary-action ``n``-player game."""
    profiles = [tuple((p >> i) & 1 for i in range(n)) for p in range(2 ** n)]
    out = []
    for prof in profiles:
        stable = True
        for i in range(n):
            alt = list(prof)
            alt[i] = 1 - alt[i]
            if payoff(tuple(alt), i) > payoff(prof, i):
                stable = False
                break
        if stable:
            out.append(prof)
    return out


_CONTRACT_STATUSES = ["inactive", "active", "fulfilled", "breached", "cancelled"]


def _contract_status(trigger: str, action: str, deadline: int, cancel: str | None,
                     events: Sequence[tuple[int, str]], now: int) -> str:
    t0 = None
    for t, name in sorted(events):
        if t > now:
            break
        if t0 is None:
            if name == trigger:
                t0 = t
            continue
        if cancel is not None and name == cancel:
            return "cancelled"
        if name == action and t <= t0 + deadline:
            return "fulfilled"
    if t0 is None:
        return "inactive"
    return "breached" if now > t0 + deadline else "active"


def _partitions(items: Sequence[str]) -> list[list[list[str]]]:
    if not items:
        return [[]]
    first, rest = items[0], list(items[1:])
    out = []
    for p in _partitions(rest):
        for i in range(len(p)):
            out.append([list(b) + [first] if j == i else list(b) for j, b in enumerate(p)])
        out.append([list(b) for b in p] + [[first]])
    return [[sorted(b) for b in sorted(p)] for p in out]
