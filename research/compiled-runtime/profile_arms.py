"""Profile attribution before and after, on the same parse and the same input.

Arm A is grouped by the roles `research/inference-cost/profile_path.py` used, so
the "before" column is directly comparable with that track's table.  Arm C is
grouped by the roles its generated code actually has: the boundary encoder, the
scalar canonicalisers that survive on internal edges, and the straight-line
operator work itself.

`cProfile` inflates absolute wall clock; the shares are the measurement.
"""
from __future__ import annotations

import cProfile
import pathlib
import pstats
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import fixtures
from harness import cached_interpreter, dump
from tcn.compile import compile_program
from tcn.types import Value

ROLES_A = {
    'Type.decode (types.py:171,175)': [('tcn/types.py', 171), ('tcn/types.py', 175)],
    'Type.encode (types.py:134,142)': [('tcn/types.py', 134), ('tcn/types.py', 142)],
    'validate_raw (types.py:185)': [('tcn/types.py', 185)],
    'numeric/type guards called by both': [(None, 'isfinite'), (None, 'isinstance'),
                                           (None, 'getattr'), (None, 'struct.unpack'),
                                           (None, 'struct.pack')],
    'Type/Value dict round-trip': [('tcn/types.py', 81), ('tcn/types.py', 89),
                                   ('dataclasses.py', None)],
    'operator semantics + graph walk': [('tcn/operators.py', None), ('tcn/graph.py', None)],
}


def group(stats, roles):
    total = stats.total_tt
    buckets = {k: 0.0 for k in roles} | {'other': 0.0}
    hot = []
    for (fn, lineno, name), (_cc, nc, tt, _ct, _) in stats.stats.items():
        placed = None
        for role, patterns in roles.items():
            for f, n in patterns:
                if f is not None and f not in fn:
                    continue
                if isinstance(n, int):
                    if lineno == n:
                        placed = role
                elif n is None or n in name:
                    placed = role
                if placed:
                    break
            if placed:
                break
        buckets[placed or 'other'] += tt
        if tt > 0.005:
            hot.append({'function': f'{fn.split("/")[-1]}:{lineno}({name})', 'calls': nc,
                        'tottime_s': tt, 'role': placed or 'other'})
    return {'profiled_total_seconds': total, 'by_role_seconds': buckets,
            'by_role_share': {k: v / total for k, v in buckets.items()},
            'hot_functions': sorted(hot, key=lambda r: -r['tottime_s'])[:20]}


def roles_c(module_name):
    return {
        'boundary encode/validate (_b*)': [(module_name, '_b')],
        'surviving scalar canonicalisers (_c*)': [(module_name, '_c')],
        'straight-line operator work': [(module_name, '_m'), (module_name, 'run'),
                                        (module_name, '<genexpr>')],
        'builtin guards (min/max/len/sorted)': [(None, 'min'), (None, 'max'), (None, 'len'),
                                                (None, 'sorted'), (None, 'round')],
    }


def main(name='visual'):
    fx = fixtures.FIXTURES[name]()
    p, r = fx['program'], fx['registry']
    types = dict(p.inputs)
    c0 = fx['cases'][0]
    ins = {k: Value.of(types[k], c0[k]) for k in types}
    res = compile_program(p, r)
    mod = res.module('genc')
    native = {k: c0[k] for k in types}

    out = {'artifact': name,
           'note': 'cProfile inflates wall clock; the shares are the measurement'}

    pr = cProfile.Profile(); pr.enable()
    p.run(ins, registry=r)
    pr.disable()
    out['arm_A'] = group(pstats.Stats(pr), ROLES_A)

    with cached_interpreter():
        ins_b = {k: Value.of(types[k], c0[k]) for k in types}
        pr = cProfile.Profile(); pr.enable()
        p.run(ins_b, registry=r)
        pr.disable()
        out['arm_B'] = group(pstats.Stats(pr), ROLES_A)

    pr = cProfile.Profile(); pr.enable()
    mod.run(native, validate=True)
    pr.disable()
    out['arm_C'] = group(pstats.Stats(pr), roles_c('genc'))

    pr = cProfile.Profile(); pr.enable()
    fx['reference'](c0)
    pr.disable()
    out['arm_D'] = group(pstats.Stats(pr), {'hand-written body': [(None, 'reference')]})

    for arm in ('arm_A', 'arm_B', 'arm_C', 'arm_D'):
        print(arm, out[arm]['profiled_total_seconds'])
        for k, v in sorted(out[arm]['by_role_share'].items(), key=lambda kv: -kv[1]):
            print(f'   {v:7.2%}  {k}')
    dump(f'profile_{name}', out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'visual')
