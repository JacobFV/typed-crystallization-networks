"""Measurement instruments shared by the four arms.

Nothing under `tcn/` or `generators/` is modified by anything in this directory.
Arm B's caches are monkeypatched for the duration of a `with` block and undone
afterwards, exactly as `research/inference-cost/visual.py` did, so the two
measurements stay comparable.

Three load-independent counters accompany every wall clock, because the host is
shared:

* `Counter`      -- applications of `Registry.exact`, one typed operator each.
* `TypeCounter`  -- calls into `tcn.types.encode`, `decode` and `validate_raw`.
                    These recurse per element, so the count is element-work, not
                    edge count: it is the quantity the compiler is supposed to
                    remove.
* `CanonCounter` -- calls into a generated module's `_c*`/`_b*` helpers, which
                    are the only encode/decode operations arm C performs.
"""
from __future__ import annotations

import contextlib
import gc
import json
import pathlib
import resource
import statistics
import time
import tracemalloc

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'out'
OUT.mkdir(parents=True, exist_ok=True)


class Counter:
    """Count typed operator applications inside a `with` block."""

    def __init__(self, registry):
        self.registry = registry
        self.n = 0

    def __enter__(self):
        self._real = self.registry.exact

        def counted(op, args):
            self.n += 1
            return self._real(op, args)
        self.registry.exact = counted
        return self

    def __exit__(self, *a):
        self.registry.exact = self._real
        return False


class TypeCounter:
    """Count `encode` / `decode` / `validate_raw` calls, recursion included."""

    def __init__(self):
        self.counts = {'encode': 0, 'decode': 0, 'validate_raw': 0}

    def __enter__(self):
        import tcn.types as TT
        self._saved = {k: getattr(TT, k) for k in self.counts}

        def wrap(key, fn):
            def g(*a, **k):
                self.counts[key] += 1
                return fn(*a, **k)
            return g
        # recursion goes through the module global, so the wrapper sees it
        for k in self.counts:
            setattr(TT, k, wrap(k, self._saved[k]))
        return self

    def __exit__(self, *a):
        import tcn.types as TT
        for k, v in self._saved.items():
            setattr(TT, k, v)
        return False

    @property
    def total(self):
        return sum(self.counts.values())


class CanonCounter:
    """Count a generated module's encode/decode helper calls."""

    def __init__(self, module):
        self.module = module
        self.n = 0

    def __enter__(self):
        self._saved = {}
        d = self.module.__dict__
        for k, v in list(d.items()):
            if (k.startswith('_c') or k.startswith('_b')) and callable(v) and k != '_cexp':
                self._saved[k] = v

                def wrap(fn):
                    def g(x):
                        self.n += 1
                        return fn(x)
                    return g
                d[k] = wrap(v)
        return self

    def __exit__(self, *a):
        self.module.__dict__.update(self._saved)
        return False


class CallCounter:
    """Count calls to named functions of a generated module."""

    def __init__(self, module, names):
        self.module = module
        self.names = list(names)
        self.counts = {n: 0 for n in self.names}

    def __enter__(self):
        d = self.module.__dict__
        self._saved = {n: d[n] for n in self.names}
        for n in self.names:
            def wrap(name, fn):
                def g(*a):
                    self.counts[name] += 1
                    return fn(*a)
                return g
            d[n] = wrap(n, self._saved[n])
        return self

    def __exit__(self, *a):
        self.module.__dict__.update(self._saved)
        return False


@contextlib.contextmanager
def cached_interpreter():
    """Arm B: memoized `Value.decoded`, no revalidation, interned `Type` load.

    Three changes, none of which alters what the interpreter computes:

    * `Value.decoded` memoizes its result on the instance, so a carrier is
      decoded once rather than once per read.
    * `Value.__post_init__` stops re-validating.  Every carrier the interpreter
      itself builds comes out of `encode`, which has already enforced the whole
      contract; only values crossing the external boundary need `validate_raw`,
      and those still get it through `Value.of`.
    * `Type.from_dict` interns, so two structurally equal types are the same
      object again and `Registry.exact`'s opening `tuple(v.type ...) != op.inputs`
      is an identity test rather than a deep compare of a 3,072-field tuple
      (`research/inference-cost/RESULTS.md` sec 4.3.1).
    """
    import tcn.types as TT
    old_post = TT.Value.__post_init__
    old_dec = TT.Value.decoded
    old_from = TT.Type.from_dict
    memo = {}

    def decoded_cached(self):
        try:
            return object.__getattribute__(self, '_dec')
        except AttributeError:
            pass
        d = TT.decode(self.type, self.raw)
        object.__setattr__(self, '_dec', d)
        return d

    def interned(cls, d):
        t = old_from.__func__(cls, d)
        key = json.dumps(t.to_dict(), sort_keys=True, separators=(',', ':'))
        return memo.setdefault(key, t)

    TT.Value.decoded = property(decoded_cached)
    TT.Value.__post_init__ = lambda self: None
    TT.Type.from_dict = classmethod(interned)
    try:
        yield memo
    finally:
        TT.Value.decoded = old_dec
        TT.Value.__post_init__ = old_post
        TT.Type.from_dict = old_from


def timed(fn, repeats=3):
    """Median and minimum wall clock in ms, gc disabled, plus the value."""
    samples = []
    gc.collect()
    gc.disable()
    try:
        value = fn()                       # warm
        for _ in range(repeats):
            t = time.perf_counter_ns()
            value = fn()
            samples.append((time.perf_counter_ns() - t) / 1e6)
    finally:
        gc.enable()
    samples.sort()
    return {'median_ms': statistics.median(samples), 'min_ms': samples[0],
            'repeats': repeats}, value


def peak_alloc(fn):
    """Peak Python allocation during one call, in MB, and the value."""
    gc.collect()
    tracemalloc.start()
    value = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6, value


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def type_identity(program, registry):
    """Distinct canonical types vs distinct `Type` objects in a live program.

    If the two numbers agree the live graph already shares one object per type
    and interning at load can only matter on the reloaded path.
    """
    seen_id, seen_key = set(), set()

    def visit(t):
        if id(t) in seen_id:
            return
        seen_id.add(id(t))
        seen_key.add(json.dumps(t.to_dict(), sort_keys=True, separators=(',', ':')))
        for x in t.items:
            visit(x)

    def walk(p):
        for _, t in p.inputs:
            visit(t)
        for _, v in p.constants:
            visit(v.type)
        for n in p.nodes:
            visit(n.output)
            for c in n.candidates:
                visit(c.operator.output)
                for t in c.operator.inputs:
                    visit(t)
                nm = c.operator.name
                if nm.startswith('module:'):
                    walk(registry.modules[nm])
                m = dict(c.operator.parameters).get('module')
                if m:
                    walk(registry.modules[m])
    walk(program)
    return {'distinct_type_objects': len(seen_id), 'distinct_canonical_types': len(seen_key)}


def dump(name, obj):
    p = OUT / f'{name}.json'
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    print(f'-> {p}')
    return p
