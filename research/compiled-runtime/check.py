"""Compile one fixture and assert exact equivalence with the interpreter.

Run before any timing is reported.  Arm C must reproduce arm A's output exactly
on every case, and arm D is checked at the same time.
"""
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import fixtures
from tcn.compile import compile_program
from tcn.types import Value


def native_out(values):
    return {k: v.decoded for k, v in values.items()}


def main(name):
    t = time.perf_counter()
    fx = fixtures.FIXTURES[name]()
    print(f'built {name} in {time.perf_counter() - t:.2f}s')
    p, r = fx['program'], fx['registry']
    print('nodes', len(p.nodes), 'modules', len(r.modules))
    t = time.perf_counter()
    res = compile_program(p, r)
    print(f'compiled in {time.perf_counter() - t:.2f}s ; stats {res.stats}')
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / f'out/{name}_compiled.py').write_text(res.source)
    m = res.module()
    types = dict(p.inputs)
    coerce = fx.get('coerce', lambda x: x)
    for i, c in enumerate(fx['cases']):
        ins = {k: Value.of(types[k], c[k]) for k in types}
        a = native_out(p.run(ins, registry=r)[0])
        cc = m.run({k: c[k] for k in types})[0]
        d = fx['reference'](c)
        assert a == cc, f'case {i}: arm C differs from arm A\n{a}\n{cc}'
        agree = all(coerce(a[k]) == coerce(d[k]) for k in d)
        print(f'case {i}: A==C exact; A vs D {"identical" if agree else "DIFFERS"}')
        if not agree:
            for k in d:
                print('   ', k, repr(a[k])[:120], '|', repr(d[k])[:120])
    print('OK')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'mixed')
