"""Why these two table pools, verified rather than asserted.

    .venv/bin/python research/nondegenerate-generalization/pools.py
"""
import itertools

DEGENERATE = (0, 3, 5, 10, 12, 15)
NONDEGENERATE = tuple(t for t in range(16) if t not in DEGENERATE)
AFFINE = (0, 3, 5, 6, 9, 10, 12, 15)          # track 4's training pool
POOL_A = (1, 2, 13, 14)
POOL_B = (4, 6, 9, 11)

def output(table, a, b):
    return bool((table >> (2 * int(a) + int(b))) & 1)

def balanced(pool):
    """Every input assignment true exactly as often as false across the pool."""
    n = len(pool)
    return n % 2 == 0 and all(sum((t >> i) & 1 for t in pool) == n // 2 for i in range(4))

def ceiling(pool):
    """Best return any fixed input-independent gate achieves, over both objectives."""
    total = len(pool) * 4 * 2
    best = max(sum((output(g, a, b) != inv) == (output(t, a, b) != inv)
                   for t in pool
                   for a, b in itertools.product((False, True), repeat=2)
                   for inv in (False, True))
               for g in range(16))
    return best / total * 4

if __name__ == '__main__':
    overlap = [t for t in AFFINE if t in DEGENERATE]
    print(f"track 4's AFFINE pool {AFFINE}")
    print(f"  degenerate members: {overlap}  ->  {len(overlap)}/{len(AFFINE)} of that pool "
          f"ignores at least one input")
    for name, pool in (('POOL_A', POOL_A), ('POOL_B', POOL_B)):
        print(f"{name} {pool}: non-degenerate={all(t in NONDEGENERATE for t in pool)} "
              f"balanced={balanced(pool)} ceiling={ceiling(pool):.2f}/4 "
              f"affine_members={[t for t in pool if t in AFFINE]}")
    assert not set(POOL_A) & set(POOL_B)
    # No balanced disjoint split separates the two affine non-degenerate tables.
    quads = [p for p in itertools.combinations(NONDEGENERATE, 4) if balanced(p)]
    split = [(x, y) for x, y in itertools.permutations(quads, 2)
             if not set(x) & set(y) and 6 in x and 9 in y]
    print(f"balanced non-degenerate 4-subsets: {len(quads)}; "
          f"disjoint pairs separating tables 6 and 9: {len(split)}")
