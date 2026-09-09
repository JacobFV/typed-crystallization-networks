"""Four arms on the same frozen program and the same inputs.

    A  the generic interpreter, as shipped
    B  a cached / interned interpreter (memoized `Value.decoded`, no
       revalidation of interpreter-produced carriers, interned `Type` at load)
    C  generated direct Python, from `tcn.compile`
    D  the same function hand-written in plain Python

Exact output equivalence is asserted against arm A for arms B and C before any
timing is recorded.  Arm D's agreement with arm A is reported, not assumed.

Every arm is measured twice: `complete` includes the external boundary (arm A
encodes its inputs with `Value.of` and decodes its outputs; arm C validates at
its boundary) and `program` is the frozen computation alone on values that are
already in the arm's own representation.  The headline C-vs-D ratio is the
`program` column, because that is the column where both arms take and return the
same native Python values.

Usage:  python research/compiled-runtime/bench.py <mixed|language|visual>
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import fixtures
from harness import (CallCounter, CanonCounter, Counter, TypeCounter, cached_interpreter,
                     dump, peak_alloc, rss_mb, timed, type_identity)
from tcn.compile import compile_program
from tcn.runtime import save_program
from tcn.types import Value

REPEATS = {'mixed': 201, 'language': 21, 'visual': 3, 'computer': 51}


def decoded(values):
    return {k: v.decoded for k, v in values.items()}


def main(name):
    t0 = time.perf_counter()
    fx = fixtures.FIXTURES[name]()
    build_s = time.perf_counter() - t0
    p, r, cases = fx['program'], fx['registry'], fx['cases']
    types = dict(p.inputs)
    reps = REPEATS[name]
    report = {'artifact': name, 'what': fx['what'], 'build_seconds': build_s,
              'cases': len(cases), 'repeats': reps}

    # ---- compile -------------------------------------------------------
    t0 = time.perf_counter()
    res = compile_program(p, r)
    report['compile_seconds'] = time.perf_counter() - t0
    report['compiler'] = res.stats
    src = HERE / f'out/{name}_compiled.py'
    src.write_text(res.source)
    mod = res.module()

    # ---- structure -----------------------------------------------------
    def static(prog):
        n = len(prog.nodes)
        ops = {c.operator.name for nd in prog.nodes for c in nd.candidates}
        return n, ops
    n_main, ops_main = static(p.pruned())
    n_all, ops_all = n_main, set(ops_main)
    for m in r.modules.values():
        k, o = static(m)
        n_all += k
        ops_all |= o
    widths = [t.width for _, t in p.inputs] + [nd.output.width for nd in p.nodes]
    for m in r.modules.values():
        widths += [nd.output.width for nd in m.nodes]
    report['structure'] = {'caller_nodes': n_main, 'total_static_nodes': n_all,
                           'distinct_operators': len(ops_all),
                           'modules': len(r.modules),
                           'widest_declared_value_elements': max(widths)}
    report['type_identity_live'] = type_identity(p, r)

    # ---- equivalence, before any timing --------------------------------
    ref_a = []
    for c in cases:
        ins = {k: Value.of(types[k], c[k]) for k in types}
        ref_a.append(decoded(p.run(ins, registry=r)[0]))
    eq = {'B_identical_to_A': True, 'C_identical_to_A': True, 'D_agreement': []}
    with cached_interpreter():
        for c, a in zip(cases, ref_a):
            ins = {k: Value.of(types[k], c[k]) for k in types}
            eq['B_identical_to_A'] &= decoded(p.run(ins, registry=r)[0]) == a
    coerce = fx.get('coerce', lambda x: x)
    has_d = fx['reference'] is not None
    for c, a in zip(cases, ref_a):
        got = mod.run({k: c[k] for k in types})[0]
        eq['C_identical_to_A'] &= got == a
        if not has_d:
            continue
        d = fx['reference'](c)
        same = all(coerce(a[k]) == coerce(d[k]) for k in d)
        err = None
        if not same:
            try:
                err = max(abs(a[k] - d[k]) for k in d)
            except TypeError:
                err = 'structurally different'
        eq['D_agreement'].append({'identical': bool(same), 'discrepancy': err})
    eq['D_identical_to_A'] = has_d and all(x['identical'] for x in eq['D_agreement'])
    report['equivalence'] = eq
    assert eq['C_identical_to_A'], 'arm C is not exactly equal to arm A -- refusing to time'
    assert eq['B_identical_to_A'], 'arm B is not exactly equal to arm A -- refusing to time'

    # ---- work counters --------------------------------------------------
    c0 = cases[0]
    ins0 = {k: Value.of(types[k], c0[k]) for k in types}
    with Counter(r) as ct:
        p.run(ins0, registry=r)
    ops_applied = ct.n
    counts = {}
    with TypeCounter() as tc:
        i = {k: Value.of(types[k], c0[k]) for k in types}
        decoded(p.run(i, registry=r)[0])
    counts['A'] = dict(tc.counts, total=tc.total)
    with cached_interpreter():
        with TypeCounter() as tc:
            i = {k: Value.of(types[k], c0[k]) for k in types}
            decoded(p.run(i, registry=r)[0])
        counts['B'] = dict(tc.counts, total=tc.total)
    native0 = {k: c0[k] for k in types}
    with CanonCounter(mod) as cc:
        mod.run(native0, validate=True)
    c_all = cc.n
    with CanonCounter(mod) as cc:
        mod.run(native0, validate=False)
    counts['C'] = {'boundary': c_all - cc.n, 'internal': cc.n, 'total': c_all}
    mfns = sorted(set(res.stats.get('module_function', {}).values()))
    per_scope = res.stats.get('emitted_per_scope', {})
    fn_of = res.stats.get('module_function', {})
    with CallCounter(mod, mfns) as mc:
        mod.run(native0, validate=False)
    c_ops = per_scope.get('run', 0)
    for mname, fn in fn_of.items():
        c_ops += mc.counts.get(fn, 0) * per_scope.get(mname, 0)
    report['work'] = {'operator_applications_A': ops_applied,
                      'operator_applications_C': c_ops,
                      'module_calls_C': dict(mc.counts),
                      'encode_decode_validate_calls': counts}

    # ---- timing ---------------------------------------------------------
    arms = {}
    enc, _ = timed(lambda: {k: Value.of(types[k], c0[k]) for k in types}, reps)
    a_prog, out_a = timed(lambda: p.run(ins0, registry=r)[0], reps)
    dec, _ = timed(lambda: decoded(out_a), reps)
    arms['A'] = {'program_ms': a_prog['median_ms'], 'program_min_ms': a_prog['min_ms'],
                 'boundary_encode_ms': enc['median_ms'], 'boundary_decode_ms': dec['median_ms'],
                 'complete_ms': enc['median_ms'] + a_prog['median_ms'] + dec['median_ms']}
    arms['A']['peak_alloc_mb'], _ = peak_alloc(lambda: decoded(p.run(
        {k: Value.of(types[k], c0[k]) for k in types}, registry=r)[0]))

    with cached_interpreter():
        enc_b, ins_b = timed(lambda: {k: Value.of(types[k], c0[k]) for k in types}, reps)
        b_prog, out_b = timed(lambda: p.run(ins_b, registry=r)[0], reps)
        dec_b, _ = timed(lambda: decoded(out_b), reps)
        arms['B'] = {'program_ms': b_prog['median_ms'], 'program_min_ms': b_prog['min_ms'],
                     'boundary_encode_ms': enc_b['median_ms'],
                     'boundary_decode_ms': dec_b['median_ms'],
                     'complete_ms': enc_b['median_ms'] + b_prog['median_ms'] + dec_b['median_ms']}
        arms['B']['peak_alloc_mb'], _ = peak_alloc(lambda: decoded(p.run(
            {k: Value.of(types[k], c0[k]) for k in types}, registry=r)[0]))

    c_prog, _ = timed(lambda: mod.run(native0, validate=False), reps)
    c_full, _ = timed(lambda: mod.run(native0, validate=True), reps)
    arms['C'] = {'program_ms': c_prog['median_ms'], 'program_min_ms': c_prog['min_ms'],
                 'boundary_encode_ms': c_full['median_ms'] - c_prog['median_ms'],
                 'boundary_decode_ms': 0.0, 'complete_ms': c_full['median_ms']}
    arms['C']['peak_alloc_mb'], _ = peak_alloc(lambda: mod.run(native0, validate=True))

    if has_d:
        d_prog, _ = timed(lambda: fx['reference'](c0), reps)
        arms['D'] = {'program_ms': d_prog['median_ms'], 'program_min_ms': d_prog['min_ms'],
                     'boundary_encode_ms': 0.0, 'boundary_decode_ms': 0.0,
                     'complete_ms': d_prog['median_ms']}
        arms['D']['peak_alloc_mb'], _ = peak_alloc(lambda: fx['reference'](c0))
    report['arms'] = arms
    report['ratios'] = {'A_over_C_program': arms['A']['program_ms'] / arms['C']['program_ms'],
                        'A_over_B_program': arms['A']['program_ms'] / arms['B']['program_ms']}
    if has_d:
        report['ratios'].update({
            'A_over_D_program': arms['A']['program_ms'] / arms['D']['program_ms'],
            'B_over_D_program': arms['B']['program_ms'] / arms['D']['program_ms'],
            'C_over_D_program': arms['C']['program_ms'] / arms['D']['program_ms'],
            'A_over_D_complete': arms['A']['complete_ms'] / arms['D']['complete_ms'],
            'C_over_D_complete': arms['C']['complete_ms'] / arms['D']['complete_ms'],
        })

    # ---- bytes on disk ---------------------------------------------------
    import gzip
    art = HERE / f'out/{name}_artifact.json'
    save_program(p, art, r)
    ab = art.read_bytes()
    interp = sum((HERE.parents[1] / 'tcn' / f).stat().st_size
                 for f in ('types.py', 'operators.py', 'graph.py'))
    cb = src.read_bytes()
    import inspect
    dtext = inspect.getsource(fx['reference']).encode() if has_d else b''
    report['bytes'] = {
        'A_artifact_json': len(ab), 'A_artifact_json_gzip': len(gzip.compress(ab, 9)),
        'A_interpreter_source': interp,
        'C_generated_python': len(cb), 'C_generated_python_gzip': len(gzip.compress(cb, 9)),
        'D_hand_written_source': len(dtext),
    }
    report['peak_rss_mb_process'] = rss_mb()
    dump(f'bench_{name}', report)
    print(json.dumps({'arms': arms, 'ratios': report['ratios'],
                      'equivalence': {k: v for k, v in eq.items() if k != 'D_agreement'}}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'mixed')
