"""What the identity guard costs when it does *not* fire.

A recogniser that fails has done work for nothing, and the honest question is how
much.  Three inputs the guard cannot accept, all of which the boundary must still
validate in full:

  list          a JSON-decoded observation.  `type(x) is tuple` is False, so the
                guard short-circuits on its first test.  This is the *deployment*
                path -- `research/compiled-runtime/RESULTS.md` section 11 sends
                every artifact through a JSON line -- so it is the case that
                decides whether the guard is worth shipping.
  float tuple   a tuple whose elements are integral floats.  `Value.of` accepts
                these and canonicalises them to `int`, so the guard must decline
                and the whole container is genuinely rebuilt.
  late reject   a tuple that is canonical except for its last element, so both
                C-level passes run to completion and then fail.
"""
from __future__ import annotations

import gc
import pathlib
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(ROOT / 'research' / 'compiled-runtime'), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import fixtures                                            # noqa: E402
from boundary import _NoGuard                              # noqa: E402
from measure import dump                                   # noqa: E402
from tcn.compile import compile_program                    # noqa: E402


def timed(fn, reps):
    gc.disable()
    try:
        s = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            s.append((time.perf_counter() - t0) * 1e3)
    finally:
        gc.enable()
    return {'median_ms': statistics.median(s), 'min_ms': min(s)}


def variants(v):
    """The same observation in each shape, keeping the outer tuple structure."""
    if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], tuple):
        head, body = v
        return {'canonical tuple': v,
                'JSON list': (head, list(body)),
                'integral floats': (head, tuple(float(x) for x in body)),
                'late reject': (head, body[:-1] + (float(body[-1]),))}
    if isinstance(v, tuple):
        return {'canonical tuple': v, 'JSON list': list(v),
                'integral floats': tuple(float(x) for x in v),
                'late reject': v[:-1] + (float(v[-1]),)}
    return {'canonical tuple': v}


def main(name, reps=201):
    fx = fixtures.FIXTURES[name]()
    p, r = fx['program'], fx['registry']
    types = dict(p.inputs)
    port = max(types, key=lambda k: types[k].width)
    c0 = {k: fx['cases'][0][k] for k in types}

    with _NoGuard():
        before = compile_program(p, r).module('before')
    after = compile_program(p, r).module('after')

    rows = []
    for label, val in variants(c0[port]).items():
        b_out = before._IN[port](val)
        a_out = after._IN[port](val)
        row = {'input': label, 'identical': b_out == a_out,
               'returned_by_reference': a_out is val,
               'before_ms': timed(lambda v=val: before._IN[port](v), reps)['median_ms'],
               'after_ms': timed(lambda v=val: after._IN[port](v), reps)['median_ms']}
        row['after_over_before'] = row['after_ms'] / row['before_ms']
        rows.append(row)
        print('%-18s identical=%-5s by-ref=%-5s before %.5f ms  after %.5f ms  x%.3f'
              % (label, row['identical'], row['returned_by_reference'],
                 row['before_ms'], row['after_ms'], row['after_over_before']))
    dump('boundary_worstcase_' + name, {'artifact': name, 'port': port,
                                        'width': types[port].width, 'rows': rows})


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'computer',
         int(sys.argv[2]) if len(sys.argv) > 2 else 201)
