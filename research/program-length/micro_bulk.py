"""Which C-level spelling of the identity guard is cheapest, and is it general?

The guard has to establish, over a homogeneous tuple, that every element is a
Python `int` (not `bool`, not `float` -- `hash(1.0) == hash(True) == hash(1)`, so
a bare membership test accepts both and would return a non-canonical value) and
that every element lies in the type's admissible interval.  Three spellings, all
sound; the interval forms carry no cardinality cap, which the frozenset does.
"""
from __future__ import annotations

import gc
import statistics
import time

_INTTYPE = frozenset((int,))
_OK = frozenset(range(256))
LO, HI = 0, 255


def superset(x):
    return type(x) is tuple and set(map(type, x)) == _INTTYPE and _OK.issuperset(x)


def minmax(x):
    return (type(x) is tuple and set(map(type, x)) == _INTTYPE
            and LO <= min(x) and max(x) <= HI)


def allcmp(x):
    return type(x) is tuple and all(type(v) is int and LO <= v <= HI for v in x)


def timed(fn, arg, reps=400):
    gc.disable()
    try:
        s = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(arg)
            s.append((time.perf_counter() - t0) * 1e3)
    finally:
        gc.enable()
    return statistics.median(s), min(s)


def main():
    for n in (4096, 3072):
        data = tuple((i * 37) % 256 for i in range(n))
        print('n =', n)
        for f in (superset, minmax, allcmp):
            assert f(data) is True
            med, mn = timed(f, data)
            print('  %-10s %8.4f ms median  %8.4f ms min' % (f.__name__, med, mn))
    # soundness on every element type that hashes equal to an admissible int
    for bad in (True, False, 1.0, 0.0, 255.0):
        d = list((i * 37) % 256 for i in range(4096))
        d[100] = bad
        t = tuple(d)
        print('bad=%-6r superset=%s minmax=%s allcmp=%s (all must be False)'
              % (bad, superset(t), minmax(t), allcmp(t)))


if __name__ == '__main__':
    main()
