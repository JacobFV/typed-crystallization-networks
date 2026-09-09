"""Exhaustive and sampled enumeration of *pruned* straight-line programs.

Same basis and conventions as `research/earned-abstraction/minimal.py` (`and`,
`or`, `xor`, `not` over BOOL, no constants, no value recomputed twice),
extended in the two ways this track needs:

*   `exhaustive(n, target, length)` returns **every** pruned straight-line
    program of exactly `length` gates whose last gate is the target -- not just
    the first in enumeration order -- with an exhaustion flag and the DFS node
    count.  A program is *pruned* when every gate other than the last is read
    by a later gate; unpruned programs of length L are pruned programs of
    length < L and are excluded so a corpus never double-counts them.

*   `sample(n, target, length, trials, seed)` draws length-`length` pruned
    conforming programs from a declared reproducible distribution: a uniform
    random (length-1)-gate prefix over the growing pool, rejecting any gate
    that recomputes a value already in the pool, then *every* single-gate
    completion of that prefix to the target.  Exhausting length 6 over four
    inputs is ~1.8e11 programs, which is why sampling is used there.  The
    distribution is declared in the pre-registration; the sample is not
    exhaustive and is never reported as such.

Canonical identity of a pruned program is the frozenset of its gate definitions
`(value, op, sorted argument values)`, which is invariant to topological
re-ordering and to gate naming.
"""
from __future__ import annotations

import random

OPS = ("and", "or", "xor")


def mask(n):
    return (1 << (2 ** n)) - 1


def var_table(n, j):
    return sum(((i >> j) & 1) << i for i in range(2 ** n))


def table_of(n, fn):
    return sum(int(bool(fn(*[bool((i >> j) & 1) for j in range(n)]))) << i
               for i in range(2 ** n))


def apply(op, u, v, m):
    if op == "and":
        return u & v
    if op == "or":
        return u | v
    if op == "xor":
        return u ^ v
    return (~u) & m


def _completions(vals, target, m):
    """Every (op, i, j) over `vals` whose result is `target`, i <= j."""
    out = []
    index = {v: i for i, v in enumerate(vals)}
    for i, u in enumerate(vals):
        j = index.get(u ^ target)
        if j is not None and j >= i:
            out.append(("xor", i, j))
    j = index.get((~target) & m)
    if j is not None:
        out.append(("not", j, j))
    sup = [i for i, u in enumerate(vals) if (u & target) == target]
    for a in range(len(sup)):
        for b in range(a, len(sup)):
            i, j = sup[a], sup[b]
            if (vals[i] & vals[j]) == target:
                out.append(("and", i, j))
    sub = [i for i, u in enumerate(vals) if (u | target) == target]
    for a in range(len(sub)):
        for b in range(a, len(sub)):
            i, j = sub[a], sub[b]
            if (vals[i] | vals[j]) == target:
                out.append(("or", i, j))
    return out


def _pruned(gates, n):
    """True when every gate other than the last is read by a later gate."""
    used = set()
    for op, i, j in gates:
        used.add(i)
        if op != "not":
            used.add(j)
    return all((n + k) in used for k in range(len(gates) - 1))


def key_of(gates, base, n):
    """Topology- and name-invariant identity of a pruned program."""
    m = mask(n)
    vals = list(base)
    defs = []
    for op, i, j in gates:
        u, v = vals[i], vals[j]
        w = apply(op, u, v, m)
        vals.append(w)
        defs.append((w, op, u, u) if op == "not" else (w, op, min(u, v), max(u, v)))
    return frozenset(defs)


def exhaustive(n, target, length, cap=None):
    """Every pruned length-`length` program for `target`.

    Returns (solutions, exhausted, dfs_nodes_expanded).  `exhausted` is False
    only when `cap` cut the collection short.
    """
    m = mask(n)
    base = [var_table(n, j) for j in range(n)]
    sols, seen = [], set()
    counter = [0]
    truncated = [False]

    def rec(vals, gates, depth):
        counter[0] += 1
        if depth == length - 1:
            if target in vals:
                return
            for g in _completions(vals, target, m):
                full = gates + [g]
                if not _pruned(full, n):
                    continue
                k = key_of(full, base, n)
                if k in seen:
                    continue
                seen.add(k)
                if cap is not None and len(sols) >= cap:
                    truncated[0] = True
                    return
                sols.append(full)
            return
        p = len(vals)
        for i in range(p):
            u = vals[i]
            for j in range(i, p):
                v = vals[j]
                for op in OPS:
                    w = apply(op, u, v, m)
                    if w in vals:
                        continue
                    rec(vals + [w], gates + [(op, i, j)], depth + 1)
            w = (~u) & m
            if w not in vals:
                rec(vals + [w], gates + [("not", i, i)], depth + 1)

    if length <= 0:
        return [], True, 0
    rec(list(base), [], 0)
    return sols, (not truncated[0]), counter[0]


def sample(n, target, length, trials, seed, cap=None):
    """Sampled pruned length-`length` programs. (solutions, trials_run, prefixes_hit)."""
    m = mask(n)
    base = [var_table(n, j) for j in range(n)]
    rng = random.Random(seed)
    all_ops = OPS + ("not",)
    sols, seen = [], set()
    hits = 0
    for t in range(trials):
        vals, gates = list(base), []
        ok = True
        for _ in range(length - 1):
            for _attempt in range(32):
                p = len(vals)
                i, j = rng.randrange(p), rng.randrange(p)
                if i > j:
                    i, j = j, i
                op = rng.choice(all_ops)
                if op == "not":
                    j = i
                w = apply(op, vals[i], vals[j], m)
                if w not in vals:
                    vals.append(w)
                    gates.append((op, i, j))
                    break
            else:
                ok = False
                break
        if not ok or target in vals:
            continue
        comps = _completions(vals, target, m)
        if comps:
            hits += 1
        for g in comps:
            full = gates + [g]
            if not _pruned(full, n):
                continue
            k = key_of(full, base, n)
            if k in seen:
                continue
            seen.add(k)
            sols.append(full)
            if cap is not None and len(sols) >= cap:
                return sols, t + 1, hits
    return sols, trials, hits
