"""Micro-benchmark: four spellings of the same 4096-element boundary check.

All four enforce the identical contract for a `uint8` tuple of arity 4096; they
differ only in how many Python frames and how much allocation they spend doing
it.  Run before changing `tcn/compile.py`, so the emitted form is chosen on a
measurement rather than on taste.
"""
from __future__ import annotations

import gc
import statistics
import time


def _nonfinite():
    raise TypeError('finite numeric value required')


def _c1(x):
    import math
    if not math.isfinite(x):
        _nonfinite()
    n = int(x)
    if n != x:
        raise ValueError('fractional value requires explicit quantization')
    if not 0 <= n <= 255:
        raise OverflowError('%s outside [0, 255]' % n)
    return n


def _b1(x):
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        _nonfinite()
    return _c1(x)


def shipped(x):
    if len(x) != 4096:
        raise ValueError('tuple arity mismatch')
    return tuple(map(_b1, x))


def genexp(x):
    if len(x) != 4096:
        raise ValueError('tuple arity mismatch')
    return tuple(v if type(v) is int and 0 <= v <= 255 else _b1(v) for v in x)


def scan_identity(x):
    if len(x) != 4096:
        raise ValueError('tuple arity mismatch')
    if type(x) is tuple and all(type(v) is int and 0 <= v <= 255 for v in x):
        return x
    return tuple(map(_b1, x))


_OK = frozenset(range(256))


def setmember(x):
    if len(x) != 4096:
        raise ValueError('tuple arity mismatch')
    if type(x) is tuple and _OK.issuperset(x):
        return x
    return tuple(map(_b1, x))


def typeset(x):
    """`setmember`'s C-level membership test, with the bool hole closed.

    `frozenset(range(256)).issuperset((True,))` is True, because `hash(True) ==
    hash(1)`, so the membership test alone silently accepts a `bool` where the
    typed boundary must raise `TypeError`.  `set(map(type, x)) == {int}` is a
    second C-level pass that closes it exactly.
    """
    if len(x) != 4096:
        raise ValueError('tuple arity mismatch')
    if type(x) is tuple and set(map(type, x)) == {int} and _OK.issuperset(x):
        return x
    return tuple(map(_b1, x))


def timed(fn, arg, reps=200):
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
    data = tuple((i * 37) % 256 for i in range(4096))
    fns = [shipped, genexp, scan_identity, setmember, typeset]
    base = shipped(data)
    print('%-16s %10s %10s  identical' % ('spelling', 'median ms', 'min ms'))
    for f in fns:
        out = f(data)
        assert tuple(out) == base, f.__name__
        med, mn = timed(f, data)
        print('%-16s %10.4f %10.4f  %s' % (f.__name__, med, mn, tuple(out) == base))
    # error contract: every spelling must raise the same thing at the same element
    for bad, want in ((0.5, ValueError), (256, OverflowError), (True, TypeError),
                      (float('inf'), TypeError), (10 ** 400, OverflowError), ('x', TypeError)):
        d = list(data)
        d[2000] = bad
        got = []
        for f in fns:
            try:
                f(tuple(d))
                got.append('no-raise')
            except Exception as e:  # noqa: BLE001
                got.append(type(e).__name__ + ': ' + str(e)[:48])
        same = len(set(got)) == 1
        print('bad=%-8.20r expected %-14s identical=%s' % (bad, want.__name__, same))
        for f, g in zip(fns, got):
            print('     %-14s %s' % (f.__name__, g))


if __name__ == '__main__':
    main()
