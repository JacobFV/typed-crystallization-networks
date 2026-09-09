"""Exact straight-line synthesis in the scaffold basis, with a certificate.

The shipped gradient path does not solve the earlier tasks (see
`out/probe_gradient_*.json`), and `tcn.search.enumerate_fit` cannot reach them
either: a 5-node, 4-input chain scaffold in the same basis admits
56*85*120*161*208 = 1.9e10 programs. So the corpus is solved exactly instead,
by iterative-deepening exhaustive search over straight-line programs in the
scaffold's own basis (`and`, `or`, `xor`, `not`) -- the same instrument the
retest's `min_program.py` uses to prove its flat minima, reimplemented here so
its node counts and exhaustion can be reported, and cross-checked against that
track's recorded figures.

The certificate this produces is genuine and is stated in the same vocabulary
as `tcn.search`: for the returned length k, every length < k was *exhausted*
with no program found, so k is the minimum in this basis. Within length k the
search returns the first program in enumeration order and stops, so it does not
claim to enumerate the conforming set at length k.

Every returned circuit is rebuilt as a `tcn.graph.Program` and executed against
the full truth table through `tcn` before it is used.
"""
from __future__ import annotations

import itertools

from tcn.types import BOOL, Value
from tcn.operators import Registry
from tcn.graph import Program, Node, Candidate

OPS = (("and", lambda u, v, m: u & v),
       ("or", lambda u, v, m: u | v),
       ("xor", lambda u, v, m: u ^ v))


def mask(n):
    return (1 << (2 ** n)) - 1


def var_table(n, j):
    return sum(((i >> j) & 1) << i for i in range(2 ** n))


def table_of(n, fn):
    return sum(int(bool(fn(*[bool((i >> j) & 1) for j in range(n)]))) << i
               for i in range(2 ** n))


def scaffold_min(n, target, max_len=6):
    """(length, gates, stats) -- the shortest straight-line program, or (None, None, stats).

    `gates` is a list of (op, i, j) indices into [inputs..., earlier gates].
    """
    m = mask(n)
    base = [var_table(n, j) for j in range(n)]
    stats = {"nodes_expanded": [], "exhausted_lengths": []}

    def rec(vals, depth, limit, counter):
        if target in vals:
            return []
        if depth == limit:
            return None
        for i, u in enumerate(vals):
            for j in range(i, len(vals)):
                v = vals[j]
                for name, fn in OPS:
                    w = fn(u, v, m)
                    if w in vals:
                        continue
                    counter[0] += 1
                    r = rec(vals + [w], depth + 1, limit, counter)
                    if r is not None:
                        return [(name, i, j)] + r
            w = (~u) & m
            if w not in vals:
                counter[0] += 1
                r = rec(vals + [w], depth + 1, limit, counter)
                if r is not None:
                    return [("not", i, i)] + r
        return None

    for limit in range(max_len + 1):
        counter = [0]
        r = rec(list(base), 0, limit, counter)
        stats["nodes_expanded"].append({"length": limit, "expanded": counter[0]})
        if r is not None:
            return limit, r, stats
        stats["exhausted_lengths"].append(limit)
    return None, None, stats


def to_program(n, gates, inputs, registry=None):
    """Rebuild a straight-line circuit as a frozen `tcn.graph.Program`."""
    r = registry or Registry()
    ports = list(inputs)
    depth = {k: 0 for k in inputs}
    nodes = []
    for k, (name, i, j) in enumerate(gates):
        out = f"g{k}"
        args = (ports[i],) if name == "not" else (ports[i], ports[j])
        op = r.resolve(name, tuple(BOOL for _ in args))
        d = max(depth[a] for a in args) + 1
        depth[out] = d
        nodes.append(Node(out, BOOL, (Candidate(op, args),), "core", d, 0))
        ports.append(out)
    return Program(tuple((k, BOOL) for k in inputs), tuple(nodes),
                   (("out", nodes[-1].name),)).validate(r)


def verify(program, fn, inputs, registry=None):
    r = registry or Registry()
    for bits in itertools.product((False, True), repeat=len(inputs)):
        out, _ = program.run({k: Value.of(BOOL, v) for k, v in zip(inputs, bits)}, registry=r)
        if bool(next(iter(out.values())).decoded) != bool(fn(*bits)):
            return False
    return True
