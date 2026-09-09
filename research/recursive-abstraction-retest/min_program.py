"""Verified minimum circuit size for every target used in the retest.

Track 5's central methodological finding was that its "9-gate" composite,
`MAJ3(a,b,c) xor MAJ3(b,c,d)`, is really a 3-gate function: overlapping windows
of a symmetric sub-function collapse algebraically, so the abstraction had
nothing to save. No target may be used here without its flat minimum being
verified rather than assumed.

Track 5's `min_program.py` does a depth-first exhaustive search over straight-line
programs in the scaffold basis (`and`, `or`, `xor`, `not`). That is exact but its
cost is roughly (branching)**depth, so it terminates only when the answer is
small -- which is why it could confirm a collapse to 3 gates but could never
confirm the *absence* of one at 6 inputs, where the branching factor is 69 and
the interesting depth is 7-9.

This module gets a rigorous lower bound a different way, so that "does not
collapse" is proved rather than searched for:

  UPPER bound  an explicit circuit, executed against the full truth table.

  LOWER bound  gate elimination over the full binary basis B2 (all sixteen
               two-input functions, negations free). If f depends on variable v
               then in any circuit v feeds at least one gate; substituting a
               constant for v makes that gate a function of its other input
               alone, so it can be deleted:

                   C(f) >= 1 + C(f | v = c)   for every live v and every c.

               Recursing on that and maximizing over (v, c) gives a lower bound
               on the number of B2 gates, and every `and`/`or`/`xor`/`not` node
               is at most one B2 gate, so it lower-bounds the *node* count of any
               flat program in the scaffold basis too.

               The recursion bottoms out at three live variables, where an
               exhaustive enumeration of all B2 circuits with at most three
               gates gives the exact minimum for every function that has one and
               a certified `>= 4` for every function that does not.

The two bounds are reported together; where they coincide the minimum is exact.
`e2_collapse_control` re-derives track 5's collapse with this machinery, so a
tool that could not see the collapse would fail visibly here.
"""
from __future__ import annotations

import itertools
import json
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).parent


# --------------------------------------------------------------------------
# packed truth tables: bit i of an n-variable table is the value at row i,
# where variable j of row i is (i >> j) & 1.
# --------------------------------------------------------------------------
def var_table(n, j):
    return sum(((i >> j) & 1) << i for i in range(2 ** n))


def table_of(n, fn):
    return sum(int(bool(fn(*[(i >> j) & 1 for j in range(n)]))) << i for i in range(2 ** n))


def mask(n):
    return (1 << (2 ** n)) - 1


def depends_on(f, n, j):
    """Does f actually depend on variable j? (exact sensitivity, not a heuristic)"""
    for i in range(2 ** n):
        if not (i >> j) & 1:
            if ((f >> i) & 1) != ((f >> (i | (1 << j))) & 1):
                return True
    return False


def live_vars(f, n):
    return tuple(j for j in range(n) if depends_on(f, n, j))


def restrict(f, n, j, c):
    """f with variable j fixed to c, as a table on n-1 variables."""
    out = 0
    idx = 0
    for i in range(2 ** n):
        if ((i >> j) & 1) != c:
            continue
        out |= ((f >> i) & 1) << idx
        idx += 1
    return out


def compress(f, n):
    """Drop the variables f does not depend on; return (table, arity)."""
    live = live_vars(f, n)
    if len(live) == n:
        return f, n
    m = len(live)
    out = 0
    for i in range(2 ** m):
        src = 0
        for k, j in enumerate(live):
            src |= ((i >> k) & 1) << j
        out |= ((f >> src) & 1) << i
    return out, m


# --------------------------------------------------------------------------
# B2: all sixteen two-input Boolean functions on packed tables
# --------------------------------------------------------------------------
def b2_apply(k, u, v, m):
    nu, nv = (~u) & m, (~v) & m
    terms = (nu & nv, nu & v, u & nv, u & v)
    out = 0
    for i, t in enumerate(terms):
        if (k >> i) & 1:
            out |= t
    return out


@lru_cache(maxsize=1)
def b2_minima_3var(kmax=3):
    """Exact B2 gate count for every 3-variable function reachable in <= kmax.

    One exhaustive depth-first enumeration of all B2 circuits of at most kmax
    gates over three inputs, deduplicating values within a circuit. Every
    function that does not appear provably needs more than kmax gates.
    """
    n, m = 3, mask(3)
    base = [var_table(n, j) for j in range(n)]
    best = {t: 0 for t in base}

    def rec(vals, depth):
        if depth == kmax:
            return
        for i in range(len(vals)):
            for j in range(i, len(vals)):
                for k in range(16):
                    w = b2_apply(k, vals[i], vals[j], m)
                    if w in vals:
                        continue
                    if best.get(w, 99) > depth + 1:
                        best[w] = depth + 1
                    rec(vals + [w], depth + 1)

    rec(list(base), 0)
    return best


def b2_lower_bound_3(f):
    """Exact if <= 3 gates suffice, otherwise the certified bound 4."""
    return b2_minima_3var().get(f, 4)


_LB_CACHE = {}


def b2_lower_bound(f, n):
    """Gate-elimination lower bound on the number of B2 gates computing f."""
    f, n = compress(f, n)
    key = (f, n)
    if key in _LB_CACHE:
        return _LB_CACHE[key]
    if n <= 1:
        out = 0
    elif n == 2:
        out = 1
    elif n == 3:
        out = b2_lower_bound_3(f)
    else:
        out = 0
        for j in range(n):
            for c in (0, 1):
                out = max(out, 1 + b2_lower_bound(restrict(f, n, j, c), n - 1))
    _LB_CACHE[key] = out
    return out


# --------------------------------------------------------------------------
# scaffold-basis exhaustive search (track 5's method), for small upper bounds
# --------------------------------------------------------------------------
SCAFFOLD_OPS = (("and", lambda u, v, m: u & v),
                ("or", lambda u, v, m: u | v),
                ("xor", lambda u, v, m: u ^ v))


def scaffold_min(n, targets, max_len=5):
    """Smallest number of and/or/xor/not nodes computing every target, or None."""
    m = mask(n)
    base = [var_table(n, j) for j in range(n)]

    def rec(vals, depth, limit):
        if all(t in vals for t in targets):
            return []
        if depth == limit:
            return None
        for i, u in enumerate(vals):
            for v in vals[i:]:
                for name, fn in SCAFFOLD_OPS:
                    w = fn(u, v, m)
                    if w in vals:
                        continue
                    r = rec(vals + [w], depth + 1, limit)
                    if r is not None:
                        return [(name, i, vals.index(v))] + r
            w = (~u) & m
            if w not in vals:
                r = rec(vals + [w], depth + 1, limit)
                if r is not None:
                    return [("not", i, i)] + r
        return None

    for limit in range(max_len + 1):
        r = rec(list(base), 0, limit)
        if r is not None:
            return limit, r
    return None, None


def verify_circuit(n, gates, target):
    """Execute an explicit and/or/xor circuit and compare against the target."""
    m = mask(n)
    vals = {f"x{j}": var_table(n, j) for j in range(n)}
    ops = {"and": lambda u, v: u & v, "or": lambda u, v: u | v,
           "xor": lambda u, v: u ^ v, "not": lambda u, v: (~u) & m}
    last = None
    for name, out, a, b in gates:
        vals[out] = ops[name](vals[a], vals[b if b is not None else a])
        last = out
    return vals[last] == target, len(gates)


# --------------------------------------------------------------------------
# the targets
# --------------------------------------------------------------------------
def maj(*xs):
    return int(sum(xs) >= 2)


TARGETS = {}


def main():
    out = {}

    # --- the sub-function: MAJ3 -------------------------------------------
    m3 = table_of(3, maj)
    s_min, s_prog = scaffold_min(3, [m3], 5)
    out["maj3_module"] = {
        "arity": 3,
        "scaffold_min_gates": s_min,
        "scaffold_program": s_prog,
        "b2_lower_bound": b2_lower_bound(m3, 3),
        "note": "sub-function to crystallize; not available as a primitive "
                "(truth_k covers every 2-input function, no 3-input one)",
    }

    # --- RETEST TARGET: disjoint windows, 6 inputs ------------------------
    # G(a,b,c,d,e,f) = MAJ3(a,b,c) xor MAJ3(d,e,f)
    g = table_of(6, lambda a, b, c, d, e, f: maj(a, b, c) ^ maj(d, e, f))
    ub_gates = [("and", "t0", "x0", "x1"), ("xor", "t1", "x0", "x1"),
                ("and", "t2", "x2", "t1"), ("or", "t3", "t0", "t2"),
                ("and", "t4", "x3", "x4"), ("xor", "t5", "x3", "x4"),
                ("and", "t6", "x5", "t5"), ("or", "t7", "t4", "t6"),
                ("xor", "y", "t3", "t7")]
    ok, size = verify_circuit(6, ub_gates, g)
    out["G_disjoint_maj3_xor"] = {
        "arity": 6, "rows": 64,
        "b2_lower_bound": b2_lower_bound(g, 6),
        "explicit_upper_bound_gates": size,
        "explicit_circuit_verified": ok,
        "all_inputs_live": len(live_vars(g, 6)) == 6,
        "abstracted_route_nodes": 3,
    }

    # --- RETEST TARGET: an AND combine on the same windows ----------------
    h = table_of(6, lambda a, b, c, d, e, f: maj(a, b, c) & maj(d, e, f))
    ub_h = ub_gates[:8] + [("and", "y", "t3", "t7")]
    okh, sizeh = verify_circuit(6, ub_h, h)
    out["H_disjoint_maj3_and"] = {
        "arity": 6, "rows": 64,
        "b2_lower_bound": b2_lower_bound(h, 6),
        "explicit_upper_bound_gates": sizeh,
        "explicit_circuit_verified": okh,
        "all_inputs_live": len(live_vars(h, 6)) == 6,
        "abstracted_route_nodes": 3,
    }

    # --- CONTROL: track 5's collapsing composite, re-derived --------------
    coll = table_of(4, lambda a, b, c, d: maj(a, b, c) ^ maj(b, c, d))
    c_min, c_prog = scaffold_min(4, [coll], 5)
    out["e2_collapse_control"] = {
        "arity": 4,
        "scaffold_min_gates": c_min, "scaffold_program": c_prog,
        "b2_lower_bound": b2_lower_bound(coll, 4),
        "note": "track 5's composite; overlapping windows collapse to (a xor d) and (b xor c)",
    }

    # --- the arm-C distractor: same interface, same verified size ---------
    # Every 3-variable function needing >= 4 B2 gates, so the control module is
    # the same size as MAJ3 by the same measure rather than by assumption.
    hard = sorted(t for t in range(256) if b2_lower_bound_3(t) >= 4)
    cands = []
    for t in hard:
        if len(live_vars(t, 3)) != 3:
            continue
        sm, sp = scaffold_min(3, [t], 5)
        if sm == 4:
            cands.append({"table": t, "scaffold_min_gates": sm, "program": sp,
                          "rows": [(t >> i) & 1 for i in range(8)]})
    # pick one that is not majority, its complement, or a variable-negation of it
    maj_family = set()
    for signs in itertools.product((0, 1), repeat=3):
        tt = table_of(3, lambda a, b, c: maj(a ^ signs[0], b ^ signs[1], c ^ signs[2]))
        maj_family |= {tt, (~tt) & mask(3)}
    # How much does the distractor actually help? Offer it as a free extra base
    # value and re-measure the minimum circuit for MAJ3. If that stays at 4, the
    # control module buys the search nothing -- which is what a control has to do.
    # Track 5's own arm-C control failed this test by inspection; here it is
    # measured. (`scaffold_min` takes the base pool from `var_table`, so the
    # extra value is spliced in by searching with an enlarged base.)
    def min_given_extra(n, targets, extra, max_len=5):
        m = mask(n)
        base = [var_table(n, j) for j in range(n)] + list(extra)

        def rec(vals, depth, limit):
            if all(t in vals for t in targets):
                return depth
            if depth == limit:
                return None
            for i, u in enumerate(vals):
                for v in vals[i:]:
                    for _, fn in SCAFFOLD_OPS:
                        w = fn(u, v, m)
                        if w in vals:
                            continue
                        r = rec(vals + [w], depth + 1, limit)
                        if r is not None:
                            return r
                w = (~u) & m
                if w not in vals:
                    r = rec(vals + [w], depth + 1, limit)
                    if r is not None:
                        return r
            return None

        for limit in range(max_len + 1):
            r = rec(list(base), 0, limit)
            if r is not None:
                return r
        return None

    scratch = min_given_extra(3, [m3], [])
    for c in cands:
        c["maj3_gates_given_this"] = min_given_extra(3, [m3], [c["table"]])
        c["in_majority_family"] = c["table"] in maj_family
    # the control must be the same verified size as MAJ3 and must not shorten it
    chosen = next(c for c in cands
                  if not c["in_majority_family"] and c["maj3_gates_given_this"] == scratch)
    out["arm_c_distractor"] = {
        "count_of_3var_functions_needing_4_or_more": len(hard),
        "count_with_scaffold_min_exactly_4_and_3_live": len(cands),
        "all_candidates": cands,
        "chosen": chosen,
        "chosen_not_in_majority_family": chosen["table"] not in maj_family,
        "b2_lower_bound": b2_lower_bound(chosen["table"], 3),
        "maj3_gates_from_scratch": scratch,
        "maj3_gates_given_distractor": chosen["maj3_gates_given_this"],
        "maj3_gates_given_maj3": min_given_extra(3, [m3], [m3]),
    }

    (HERE / "min_program.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
