"""A random differential sweep of the compiled mixed program, plus its errors.

`tests/test_compile.py` covers the operator families; this covers one *shipped*
artifact densely, on inputs it was never measured on, and checks that the error
contract fires at the same edge with the same exception type in both arms.
"""
import math
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tcn.compile import compile_program
from tcn.runtime import load_program
from tcn.types import Value

p, r = load_program(ROOT / 'artifacts/demo/synthesis/program.json')
mod = compile_program(p, r).module()
types = dict(p.inputs)
rng = random.Random(0)
bad = 0
for _ in range(2000):
    a, b = rng.random() < .5, rng.random() < .5
    x = rng.uniform(-3, 3)
    got = mod.run({'a': a, 'b': b, 'x': x})[0]['answer']
    ref = p.run({k: Value.of(types[k], v) for k, v in dict(a=a, b=b, x=x).items()},
                registry=r)[0]['answer'].decoded
    if got != ref:
        bad += 1
        print('MISMATCH', a, b, x, got, ref)
print('random cases: 2000, mismatches:', bad)

for x in (1e39, float('inf'), 1e300, 3.4e38):
    for arm, fn in (('interpreter', lambda x=x: p.run(
                        {k: Value.of(types[k], v) for k, v in
                         dict(a=True, b=False, x=x).items()}, registry=r)),
                    ('compiled', lambda x=x: mod.run({'a': True, 'b': False, 'x': x}))):
        try:
            fn()
            print(f'{x!r:12} {arm:12} ok')
        except Exception as e:
            print(f'{x!r:12} {arm:12} {type(e).__name__}: {e}')
