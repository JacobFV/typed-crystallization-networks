"""Exhaustive minimum straight-line program length for the experiment targets.

The difficulty claims in RESULTS.md ("the flat arm needs N gates") must be
*measured*, not assumed: a composite built out of two copies of a sub-function
can collapse algebraically into something much smaller, in which case the
abstraction has nothing to save.

Search space: straight-line programs over the same vocabulary the scaffolds use
(`and`, `or`, `xor` binary; `not` unary), each step reading any earlier value or
input, with full sharing. Truth tables are packed into ints so each candidate is
one machine word.
"""
from __future__ import annotations

import itertools
import json
import sys

OPS = (("and", lambda u, v, m: u & v),
       ("or", lambda u, v, m: u | v),
       ("xor", lambda u, v, m: u ^ v))


def input_tables(n):
    """Column j of the truth table for input j, as an int of 2**n bits."""
    rows = list(itertools.product((0, 1), repeat=n))
    return [sum(r[j] << i for i, r in enumerate(rows)) for j in range(n)]


def search(n, targets, max_len=6):
    """Smallest L such that some length-L program computes every target.

    `targets` is a list of packed truth tables; a program satisfies them if each
    target appears among the inputs or the computed values.
    """
    mask = (1 << (2 ** n)) - 1
    base = input_tables(n)

    def rec(vals, depth, limit):
        if all(t in vals for t in targets):
            return []
        if depth == limit:
            return None
        for i, u in enumerate(vals):
            for v in vals[i:]:
                for name, f in OPS:
                    w = f(u, v, mask)
                    if w in vals:
                        continue
                    r = rec(vals + [w], depth + 1, limit)
                    if r is not None:
                        return [(name, i, vals.index(v))] + r
            w = (~u) & mask
            if w not in vals:
                r = rec(vals + [w], depth + 1, limit)
                if r is not None:
                    return [("not", i, i)] + r
        return None

    for limit in range(0, max_len + 1):
        r = rec(list(base), 0, limit)
        if r is not None:
            return limit, r
    return None, None


def main():
    out = {}
    rows3 = list(itertools.product((0, 1), repeat=3))
    rows4 = list(itertools.product((0, 1), repeat=4))

    # E1: full adder (a, b, cin) -> (sum, carry); both outputs must appear.
    fa_sum = sum(((a + b + c) % 2) << i for i, (a, b, c) in enumerate(rows3))
    fa_carry = sum((1 if a + b + c >= 2 else 0) << i for i, (a, b, c) in enumerate(rows3))
    out["e1_full_adder"] = dict(zip(("min_gates", "program"), search(3, [fa_sum, fa_carry], 6)))

    # the half adder itself (the module learned in stage 1)
    ha_sum = sum(((a ^ b)) << i for i, (a, b) in enumerate(itertools.product((0, 1), repeat=2)))
    ha_carry = sum(((a & b)) << i for i, (a, b) in enumerate(itertools.product((0, 1), repeat=2)))
    out["e1_half_adder_module"] = dict(zip(("min_gates", "program"), search(2, [ha_sum, ha_carry], 4)))

    # E2: MAJ3(a,b,c) xor MAJ3(b,c,d)
    def maj(*xs):
        return 1 if sum(xs) >= 2 else 0
    h = sum((maj(a, b, c) ^ maj(b, c, d)) << i for i, (a, b, c, d) in enumerate(rows4))
    out["e2_sliding_majority"] = dict(zip(("min_gates", "program"), search(4, [h], 5)))

    # MAJ3 itself (the module learned in E2 stage 1)
    m3 = sum(maj(a, b, c) << i for i, (a, b, c) in enumerate(rows3))
    out["e2_maj3_module"] = dict(zip(("min_gates", "program"), search(3, [m3], 5)))

    # a non-collapsing alternative on the same 4 inputs, for reference
    hand = sum((maj(a, b, c) ^ maj(a, b, d)) << i for i, (a, b, c, d) in enumerate(rows4))
    out["e2_alt_shared_pair"] = dict(zip(("min_gates", "program"), search(4, [hand], 5)))

    print(json.dumps(out, indent=2))
    with open(sys.argv[1] if len(sys.argv) > 1 else "min_program.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
