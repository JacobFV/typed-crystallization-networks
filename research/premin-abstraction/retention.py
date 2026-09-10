"""How much of the reusable structure each corpus retains.

Two different questions, both reported, because they answer different halves of
the mechanism:

*   **value retention** -- does the program hold a node whose *value* is the
    fragment, over **any ordered 3-subset of the four inputs**?  This is the
    measurement §44 made; the naive check against `(a,b,c)` only is unfair to
    tasks like `t3_maj_bcd`, which computes a majority of a different triple.

*   **structural retention** -- is there a *single canonical abstraction*, in
    the rule's own R2 sense (same `Program.digest`), that computes majority and
    is legal at that node?  A frequency rule cannot rank a function; it ranks
    digests.  Two programs can both hold `maj` and still offer the rule nothing
    if they realise it with different circuits.
"""
from __future__ import annotations

import itertools

from tcn.operators import Registry
from tcn.types import BOOL, Value

import mine
import enumerate_programs as E

N = 4


def maj3(a, b, c):
    return (int(a) + int(b) + int(c)) >= 2


def frag_M(a, b, c):
    """§44's rank-1 proposal M(x0,x1,x2) = x2 or (x0 and x1)."""
    return bool(c) or (bool(a) and bool(b))


def frag_Mprime(a, b, c):
    """§44's runner-up M'(x0,x1,x2) = x2 and (x0 or x1)."""
    return bool(c) and (bool(a) or bool(b))


ORDERED_TRIPLES = tuple(itertools.permutations(range(N), 3))


def target_tables(fn3):
    """The 16-bit table of `fn3` applied to every ordered 3-subset of 4 inputs."""
    out = {}
    for t in ORDERED_TRIPLES:
        table = 0
        for row in range(2 ** N):
            bits = [bool((row >> j) & 1) for j in range(N)]
            if fn3(bits[t[0]], bits[t[1]], bits[t[2]]):
                table |= 1 << row
        out[t] = table
    return out


def node_values(gates):
    base = [E.var_table(N, j) for j in range(N)]
    m = E.mask(N)
    vals = list(base)
    for op, i, j in gates:
        vals.append(E.apply(op, vals[i], vals[j], m))
    return vals[N:]


def value_retention(gate_lists, fn3):
    """(programs retaining a body of `fn3`, total programs, per-triple counts)."""
    tables = target_tables(fn3)
    inv = {}
    for t, tab in tables.items():
        inv.setdefault(tab, []).append(t)
    hit, per_triple = 0, {}
    for gates in gate_lists:
        vals = set(node_values(gates))
        found = [t for tab, ts in inv.items() if tab in vals for t in ts]
        if found:
            hit += 1
            for t in found:
                per_triple[t] = per_triple.get(t, 0) + 1
    return hit, len(gate_lists), {"".join("abcd"[i] for i in k): v
                                  for k, v in sorted(per_triple.items())}


def fragment_functions(corpus, max_nodes=mine.MAX_NODES, max_holes=mine.MAX_HOLES):
    """digest -> {'arity', 'table', 'entries', 'tasks', 'nodes'} over a tcn corpus.

    `corpus` maps entry id -> (task, frozen Program).
    """
    cache, info = {}, {}
    for entry, (task, program) in corpus.items():
        seen = set()
        for _root, _S, canon, _holes in mine.fragments(program, max_nodes, max_holes):
            d = canon.digest
            if d not in cache:
                cache[d] = _table3(canon) if len(canon.inputs) == 3 else None
            rec = info.setdefault(d, {"arity": len(canon.inputs), "table": cache[d],
                                      "nodes": len(canon.nodes), "entries": 0,
                                      "tasks": set(), "sites": 0})
            rec["sites"] += 1
            rec["tasks"].add(task)
            if d not in seen:
                seen.add(d)
                rec["entries"] += 1
    return info


def _table3(canon):
    r = Registry()
    bits_out = []
    for bits in itertools.product((False, True), repeat=3):
        out, _ = canon.run({k: Value.of(BOOL, v) for (k, _), v in zip(canon.inputs, bits)},
                           registry=r)
        bits_out.append(bool(next(iter(out.values())).decoded))
    return tuple(bits_out)


MAJ3_TABLE = tuple(maj3(*b) for b in itertools.product((False, True), repeat=3))
M_TABLE = tuple(frag_M(*b) for b in itertools.product((False, True), repeat=3))
MPRIME_TABLE = tuple(frag_Mprime(*b) for b in itertools.product((False, True), repeat=3))
