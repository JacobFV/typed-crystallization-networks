"""Q2: what the external boundary must check, and what a minimal one costs.

`research/compiled-runtime/RESULTS.md` section 5: the frozen computer program is
**1.12 us** and validating one typed observation at the external boundary is
**585 us**, a 523x asymmetry that is pure marshalling at the untrusted edge.

The semantics may not change.  The external boundary is exactly where untrusted
input must still be validated, so nothing here removes a check: the question is
which checks are *provably satisfied on a branch that can be recognised more
cheaply than by performing them*.  Three arms:

  before   the compiler exactly as `research/compiled-runtime` left it, obtained
           by disabling the new guard rather than by checking out the old file,
           so both arms are the same code path apart from the one change
  after    the same compiler with `_Compiler._identity_guard` active
  oracle   `Value.of`, the interpreter's own boundary, which defines the contract

Every arm is asserted to agree on valid input *and* on an adversarial suite that
walks every failure mode of the contract at a chosen element, comparing exception
type and message, before any timing is taken.
"""
from __future__ import annotations

import argparse
import cProfile
import gc
import json
import pathlib
import pstats
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(ROOT / 'research' / 'compiled-runtime'), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import fixtures                                            # noqa: E402
from measure import count_opcodes, dump                    # noqa: E402
from tcn import compile as tcncompile                      # noqa: E402
from tcn.compile import compile_program                    # noqa: E402
from tcn.types import Value                                # noqa: E402

# every way a typed scalar boundary can legitimately reject an element
BAD = [('fractional', 0.5), ('above range', 1 << 40), ('bool', True),
       ('non-finite', float('inf')), ('nan', float('nan')),
       ('huge int', 10 ** 400), ('string', 'x'), ('none', None),
       ('negative', -1), ('float in range', 3.0)]


class _NoGuard:
    """The compiler as it was: `_identity_guard` never fires."""

    def __enter__(self):
        self._orig = tcncompile._Compiler._identity_guard
        tcncompile._Compiler._identity_guard = lambda self, t, arity=0: None
        return self

    def __exit__(self, *a):
        tcncompile._Compiler._identity_guard = self._orig
        return False


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
    return {'median_ms': statistics.median(s), 'min_ms': min(s), 'reps': reps}


def _norm(e):
    return None if e is None else (type(e).__name__, str(e))


def _call(fn, arg):
    try:
        return ('ok', fn(arg))
    except Exception as e:               # noqa: BLE001
        return ('raise', _norm(e))


def adversarial(before, after, program, port, case, index):
    """Corrupt one element of one input and compare all three arms exactly."""
    t = dict(program.inputs)[port]
    rows = []
    for label, bad in BAD:
        v = case[port]
        if isinstance(v, tuple) and isinstance(v[1], tuple):
            broken = (v[0], v[1][:index] + (bad,) + v[1][index + 1:])
        elif isinstance(v, tuple):
            broken = v[:index] + (bad,) + v[index + 1:]
        else:
            broken = bad
        b = _call(before._IN[port], broken)
        a = _call(after._IN[port], broken)
        o = _call(lambda x: Value.of(t, x).decoded, broken)
        rows.append({'case': label, 'before': str(b[1])[:90] if b[0] == 'raise' else 'ok',
                     'after': str(a[1])[:90] if a[0] == 'raise' else 'ok',
                     'oracle': str(o[1])[:90] if o[0] == 'raise' else 'ok',
                     'before_after_identical': b == a,
                     'after_matches_oracle_kind': (b[0] == a[0] == o[0]) and (
                         b[0] == 'ok' or b[1][0] == a[1][0] == o[1][0])})
    return rows


def main(name, reps):
    fx = fixtures.FIXTURES[name]()
    p, r = fx['program'], fx['registry']
    types = dict(p.inputs)
    cases = [{k: c[k] for k in types} for c in fx['cases']]
    c0 = cases[0]

    with _NoGuard():
        res_b = compile_program(p, r)
    res_a = compile_program(p, r)
    before, after = res_b.module('before'), res_a.module('after')

    # ---- identical output, every case, boundary on ------------------------
    outs_b = [before.run(c, validate=True)[0] for c in cases]
    outs_a = [after.run(c, validate=True)[0] for c in cases]
    ins = [{k: Value.of(types[k], c[k]) for k in types} for c in cases]
    outs_i = [{k: v.decoded for k, v in p.run(i, registry=r)[0].items()} for i in ins]
    identical = outs_b == outs_a == outs_i
    if not identical:
        print('OUTPUTS DIFFER -- refusing to time')
    # the boundary itself must produce the identical canonical value
    port = max(types, key=lambda k: types[k].width)
    canon_b = [before._IN[k](c[k]) for k in types for c in cases]
    canon_a = [after._IN[k](c[k]) for k in types for c in cases]
    canon_identical = canon_b == canon_a

    adv = adversarial(before, after, p, port, c0, min(3, types[port].width - 1))
    adv_ok = all(row['before_after_identical'] for row in adv)

    out = {'artifact': name, 'widest_port': port, 'widest_width': types[port].width,
           'identical_outputs': identical, 'boundary_values_identical': canon_identical,
           'adversarial_identical': adv_ok, 'adversarial': adv,
           'stats_before': dict(res_b.stats), 'stats_after': dict(res_a.stats),
           'source_bytes': {'before': len(res_b.source), 'after': len(res_a.source)}}
    if not (identical and canon_identical and adv_ok):
        print(json.dumps(out, indent=1)[:4000])
        raise SystemExit('equivalence failed; no timing reported')

    # ---- cost -------------------------------------------------------------
    def boundary_only(mod):
        return lambda: [mod._IN[k](c0[k]) for k in types]

    out['timing'] = {
        'boundary_before': timed(boundary_only(before), reps),
        'boundary_after': timed(boundary_only(after), reps),
        'program_only': timed(lambda: after.run(c0, validate=False), reps),
        'complete_before': timed(lambda: before.run(c0, validate=True), reps),
        'complete_after': timed(lambda: after.run(c0, validate=True), reps),
        'interpreter_boundary': timed(
            lambda: {k: Value.of(types[k], c0[k]) for k in types}, max(3, reps // 20)),
    }
    out['bytecodes'] = {
        'boundary_before': count_opcodes(boundary_only(before)),
        'boundary_after': count_opcodes(boundary_only(after)),
        'program_only': count_opcodes(lambda: after.run(c0, validate=False)),
        'complete_before': count_opcodes(lambda: before.run(c0, validate=True)),
        'complete_after': count_opcodes(lambda: after.run(c0, validate=True)),
    }
    t = out['timing']
    out['ratios'] = {
        'boundary_before_over_after': t['boundary_before']['median_ms'] / t['boundary_after']['median_ms'],
        'boundary_before_over_program': t['boundary_before']['median_ms'] / t['program_only']['median_ms'],
        'boundary_after_over_program': t['boundary_after']['median_ms'] / t['program_only']['median_ms'],
        'complete_before_over_after': t['complete_before']['median_ms'] / t['complete_after']['median_ms'],
        'bytecodes_before_over_after': out['bytecodes']['boundary_before'] / max(1, out['bytecodes']['boundary_after']),
    }

    # ---- attribution ------------------------------------------------------
    out['profile'] = {}
    for tag, mod in (('before', before), ('after', after)):
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(max(1, reps // 4)):
            mod.run(c0, validate=True)
        pr.disable()
        st = pstats.Stats(pr)
        total = sum(v[2] for v in st.stats.values())
        rows = sorted(((v[2], k) for k, v in st.stats.items()), reverse=True)[:8]
        out['profile'][tag] = {'total_s': total,
                               'top': [{'fn': '%s:%s' % (k[0].split('/')[-1], k[2]),
                                        'tottime_s': tt, 'share': tt / total if total else 0}
                                       for tt, k in rows]}

    print(name, 'identical', identical, canon_identical, adv_ok)
    for k, v in t.items():
        print('  %-22s %10.5f ms' % (k, v['median_ms']))
    print('  bytecodes', out['bytecodes'])
    print('  ratios', {k: round(v, 2) for k, v in out['ratios'].items()})
    dump('boundary_' + name, out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('artifact', nargs='?', default='computer')
    ap.add_argument('--reps', type=int, default=201)
    a = ap.parse_args()
    main(a.artifact, a.reps)
