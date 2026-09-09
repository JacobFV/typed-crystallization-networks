"""What the external boundary actually checks, per artifact, printed verbatim.

Q2 starts from `research/compiled-runtime/RESULTS.md` section 5: the frozen
computer program runs in 1.12 us while validating one typed observation at the
external boundary costs 585 us.  Before anything is optimized, the checks have
to be enumerated and each one attributed to the contract it enforces.
"""
from __future__ import annotations

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(ROOT / 'research' / 'compiled-runtime')):
    if p not in sys.path:
        sys.path.insert(0, p)

import fixtures                                    # noqa: E402
from tcn.compile import compile_program            # noqa: E402


def main(name):
    fx = fixtures.FIXTURES[name]()
    res = compile_program(fx['program'], fx['registry'])
    src = res.source
    print('=== inputs and their boundary encoders ===')
    for line in src.splitlines():
        if line.startswith('_IN = ') or line.startswith('INPUTS = '):
            print(line[:400])
    print()
    print('=== boundary (_b*) and canonicaliser (_c*) helper bodies ===')
    for m in re.finditer(r'^def (_[bc]\d+)\(x\):\n((?:    .*\n)+)', src, re.M):
        print('def %s(x):' % m.group(1))
        print(m.group(2).rstrip('\n'))
        print()
    print('=== declared input types ===')
    for k, t in fx['program'].inputs:
        print(' ', k, 'width', t.width, 'kind', t.kind)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'computer')
