"""`tcn.compile` is a pure addition, so the interpreter is its oracle.

Every test here asserts the generated Python agrees with `Registry.exact` and
`Program.execute` *exactly* -- the same value, and the same exception at the same
edge -- rather than approximately.
"""
import math

import pytest

from tcn.compile import compile_program
from tcn.graph import Candidate, Node, Program
from tcn.operators import Registry
from tcn.types import (BOOL, Value, fixed, floating, integer, product, setof)

R = Registry()
I8 = integer(8)
U8 = integer(8, signed=False)
I16 = integer(16)
F32 = floating()
F64 = floating(64)


def build(inputs, spec, outputs, constants=(), state=(), input_depths=(), registry=None):
    """One frozen node per entry of `spec`: (name, operator, sources, params, out)."""
    r = registry or R
    ports = dict(inputs) | {k: v.type for k, v in constants} | {k: v.type for k, v, _ in state}
    nodes = []
    for depth, (name, op, sources, params, out) in enumerate(spec, start=1):
        types = tuple(ports[s] for s in sources)
        operator = r.resolve(op, types, out, params)
        nodes.append(Node(name, operator.output, (Candidate(operator, tuple(sources)),),
                          'core', depth, 0))
        ports[name] = operator.output
    p = Program(tuple(inputs), tuple(nodes), tuple(outputs), tuple(constants), tuple(state),
                tuple(input_depths))
    return p.validate(r), r


def agree(program, registry, cases, out_keys=None):
    res = compile_program(program, registry)
    mod = res.module()
    types = dict(program.inputs)
    for c in cases:
        ins = {k: Value.of(types[k], c[k]) for k in types}
        try:
            want = {k: v.decoded for k, v in program.run(ins, registry=registry)[0].items()}
            err = None
        except Exception as e:                      # the error contract is part of the semantics
            want, err = None, (type(e), str(e))
        try:
            got = mod.run(dict(c))[0]
            gerr = None
        except Exception as e:
            got, gerr = None, (type(e), str(e))
        assert (err is None) == (gerr is None), f'{c}: {err} vs {gerr}'
        if err is not None:
            assert err[0] is gerr[0], f'{c}: {err} vs {gerr}'
        else:
            assert got == want, f'{c}: {got} != {want}'
    return res, mod


def test_boolean_and_truth_tables():
    spec = [(f'g{t}', f'truth_{t}', ('a', 'b'), None, None) for t in range(16)]
    spec.append(('n', 'not', ('g0',), None, None))
    p, r = build((('a', BOOL), ('b', BOOL)), spec,
                 tuple((f'o{t}', f'g{t}') for t in range(16)) + (('on', 'n'),))
    agree(p, r, [{'a': a, 'b': b} for a in (False, True) for b in (False, True)])


def test_logic_compare_and_mux():
    spec = [('x', 'lt', ('u', 'v'), None, None),
            ('y', 'eq', ('u', 'v'), None, None),
            ('z', 'and', ('x', 'y'), None, None),
            ('w', 'mux', ('z', 'u', 'v'), None, None)]
    p, r = build((('u', I8), ('v', I8)), spec, (('z', 'z'), ('w', 'w')))
    agree(p, r, [{'u': a, 'v': b} for a in (-3, 0, 5) for b in (-3, 0, 5)])


def test_integer_arithmetic_and_overflow_contract():
    spec = [('s', 'add', ('a', 'b'), None, None),
            ('m', 'mul', ('a', 'b'), None, None)]
    p, r = build((('a', I8), ('b', I8)), spec, (('s', 's'), ('m', 'm')))
    agree(p, r, [{'a': a, 'b': b} for a in (-128, -1, 0, 3, 127) for b in (-128, -1, 0, 3, 127)])


@pytest.mark.parametrize('overflow', ['error', 'wrap', 'saturate'])
def test_overflow_policies_round_trip_identically(overflow):
    t = integer(8, overflow=overflow)
    p, r = build((('a', t), ('b', t)), [('s', 'add', ('a', 'b'), None, None)], (('s', 's'),))
    agree(p, r, [{'a': a, 'b': b} for a in (-128, -70, 0, 70, 127) for b in (-128, -70, 0, 70, 127)])


def test_zero_denominator_and_shift_bounds():
    p, r = build((('a', I8), ('b', I8)),
                 [('q', 'idiv', ('a', 'b'), None, None), ('m', 'mod', ('a', 'b'), None, None)],
                 (('q', 'q'), ('m', 'm')))
    agree(p, r, [{'a': 7, 'b': b} for b in (-2, 0, 2)])
    p, r = build((('a', I8), ('b', I8)), [('s', 'shl', ('a', 'b'), None, None)], (('s', 's'),))
    agree(p, r, [{'a': 1, 'b': b} for b in (0, 3, 8, -1)])


def test_fixed_and_float_encodings_quantize_the_same_way():
    for t in (fixed(16, 256), F32, F64):
        p, r = build((('a', t), ('b', t)),
                     [('s', 'add', ('a', 'b'), None, None),
                      ('d', 'div', ('s', 'b'), None, None)] if t is not fixed else
                     [('s', 'add', ('a', 'b'), None, None)],
                     (('s', 's'),) if t is fixed else (('s', 's'), ('d', 'd')))
        agree(p, r, [{'a': a, 'b': b} for a in (0.1, -2.75, 3.0) for b in (0.5, -0.25, 0.0)])


def test_analytic_operators():
    spec = [('e', 'exp', ('x',), None, None), ('l', 'log', ('x',), None, None),
            ('s', 'sin', ('x',), None, None), ('c', 'cos', ('x',), None, None),
            ('q', 'sqrt', ('x',), None, None), ('n', 'neg', ('x',), None, None),
            ('ab', 'abs', ('x',), None, None), ('at', 'atan2', ('x', 'y'), None, None),
            ('pw', 'pow', ('x', 'y'), None, None), ('fb', 'fourier_basis', ('x',), None, None)]
    p, r = build((('x', F64), ('y', F64)), spec, tuple((n, n) for n, *_ in spec))
    agree(p, r, [{'x': x, 'y': 1.5} for x in (0.5, 2.0, -1.0, 0.0)])


def test_bounds_are_enforced_at_the_same_edge():
    t = floating(64, bounds=(-1.0, 1.0))
    p, r = build((('a', t), ('b', t)), [('s', 'add', ('a', 'b'), None, None)], (('s', 's'),))
    agree(p, r, [{'a': a, 'b': b} for a in (-1.0, 0.25, 1.0) for b in (-1.0, 0.9, 1.0)])


def test_tuples_projection_and_dynamic_index():
    T = product(I8, I8, I8)
    spec = [('t', 'tuple', ('a', 'b', 'c'), None, None),
            ('p0', 'project', ('t',), {'index': 0}, None),
            ('ix', 'index', ('t', 'i'), None, None)]
    p, r = build((('a', I8), ('b', I8), ('c', I8), ('i', U8)), spec, (('p0', 'p0'), ('ix', 'ix')))
    agree(p, r, [{'a': 1, 'b': 2, 'c': 3, 'i': i} for i in (0, 1, 2, 3, 200)])


def test_set_algebra_and_capacity_contract():
    S = setof(I8, 3)
    spec = [('ins', 'insert', ('s', 'x'), None, None),
            ('rem', 'remove', ('ins', 'x'), None, None),
            ('un', 'union', ('ins', 's'), None, None),
            ('it', 'intersection', ('ins', 's'), None, None),
            ('mem', 'member', ('s', 'x'), None, None),
            ('n', 'count', ('s',), None, None)]
    p, r = build((('s', S), ('x', I8)), spec,
                 (('ins', 'ins'), ('rem', 'rem'), ('un', 'un'), ('it', 'it'),
                  ('mem', 'mem'), ('n', 'n')))
    agree(p, r, [{'s': (1, 2), 'x': 3}, {'s': (1, 2, 3), 'x': 4}, {'s': (), 'x': 1},
                 {'s': (1, 2, 3), 'x': 1}])


def test_reductions_over_tuple_and_set():
    T = product(F64, F64, F64)
    S = setof(F64, 4)
    spec = [('s1', 'sum', ('t',), None, None), ('mn', 'reduce_min', ('t',), None, None),
            ('mx', 'reduce_max', ('t',), None, None), ('av', 'mean', ('t',), None, None),
            ('s2', 'sum', ('u',), None, None), ('c', 'count', ('u',), None, None)]
    p, r = build((('t', T), ('u', S)), spec, tuple((n, n) for n, *_ in spec))
    agree(p, r, [{'t': (1.0, 2.0, 3.0), 'u': (1.0, 2.0)}, {'t': (0.0, 0.0, 0.0), 'u': ()}])


def test_empty_reduction_raises_in_both_arms():
    S = setof(F64, 4)
    p, r = build((('u', S),), [('mn', 'reduce_min', ('u',), None, None)], (('mn', 'mn'),))
    agree(p, r, [{'u': ()}, {'u': (1.0,)}])


def test_pack_unpack_and_interpret():
    T = product(U8, U8, BOOL)
    K = integer(17, signed=False)
    BYTE = integer(8, signed=False, role='byte')
    CAT = integer(8, signed=False, role='category')
    spec = [('t', 'tuple', ('a', 'b', 'f'), None, None),
            ('k', 'pack', ('t',), None, K),
            ('u', 'unpack', ('k',), None, T),
            ('i', 'interpret', ('z',), None, CAT)]
    p, r = build((('a', U8), ('b', U8), ('f', BOOL), ('z', BYTE)), spec,
                 (('k', 'k'), ('u', 'u'), ('i', 'i')))
    agree(p, r, [{'a': a, 'b': 200, 'f': True, 'z': 7} for a in (0, 1, 255)])


def test_signed_pack_round_trip():
    T = product(I8, I8)
    K = integer(16, signed=False)
    spec = [('t', 'tuple', ('a', 'b'), None, None), ('k', 'pack', ('t',), None, K),
            ('u', 'unpack', ('k',), None, T)]
    p, r = build((('a', I8), ('b', I8)), spec, (('k', 'k'), ('u', 'u')))
    agree(p, r, [{'a': a, 'b': b} for a in (-128, -1, 0, 127) for b in (-128, -1, 0, 127)])


def test_conversions_between_representations():
    spec = [('q', 'quantize', ('x',), None, I16),
            ('d', 'dequantize', ('q',), None, F64),
            ('e', 'encode', ('t',), None, F64),
            ('b', 'decode', ('x',), None, BOOL)]
    p, r = build((('x', F64), ('t', BOOL)), spec, tuple((n, n) for n, *_ in spec))
    agree(p, r, [{'x': x, 't': t} for x in (0.4, 0.6, -2.5, 3.5) for t in (False, True)])


def test_pair_and_join():
    A = setof(product(U8, U8), 3)
    B = setof(product(U8, U8), 3)
    spec = [('pr', 'pair', ('a', 'b'), None, None),
            ('jn', 'join', ('a', 'b'), {'left': 0, 'right': 1}, None)]
    p, r = build((('a', A), ('b', B)), spec, (('pr', 'pr'), ('jn', 'jn')))
    agree(p, r, [{'a': ((1, 2), (3, 4)), 'b': ((5, 1), (6, 3))}, {'a': (), 'b': ((5, 1),)}])


def test_spectral_operators():
    T = product(*(F64 for _ in range(4)))
    spec = [('z', 'fft', ('t',), None, None), ('iz', 'ifft', ('z',), None, None)]
    p, r = build((('t', T),), spec, (('z', 'z'), ('iz', 'iz')))
    agree(p, r, [{'t': (1.0, 0.0, -1.0, 0.5)}])


def test_temporal_operators_and_state():
    r = Registry()
    spec = [('dl', 'delay', ('x', 'mem'), None, None),
            ('df', 'difference', ('x', 'mem'), None, None),
            ('ac', 'accumulation', ('x', 'mem'), None, None),
            ('nx', 'project', ('dl',), {'index': 1}, None)]
    p, r = build((('x', I16),), spec, (('dl', 'dl'), ('df', 'df'), ('ac', 'ac')),
                 state=(('mem', Value.of(I16, 0), 'nx'),), registry=r)
    res = compile_program(p, r)
    mod = res.module()
    state_i, state_c = None, None
    for x in (3, 5, -2, 7):
        out_i, state_i = p.run({'x': Value.of(I16, x)}, state_i, registry=r)
        out_c, state_c = mod.run({'x': x}, state_c)
        assert {k: v.decoded for k, v in out_i.items()} == out_c
        state_i = {k: v for k, v in state_i.items()}
        state_c = dict(state_c)


def test_modules_are_inlined_but_stay_observable():
    r = Registry()
    inner, _ = build((('u', I8), ('v', I8)),
                     [('s', 'add', ('u', 'v'), None, None)], (('y', 's'),), registry=r)
    name = r.register_module(inner)
    outer, _ = build((('a', I8), ('b', I8)),
                     [('m', name, ('a', 'b'), None, None),
                      ('n', name, ('m', 'b'), None, None)], (('y', 'n'),), registry=r)
    res, mod = agree(outer, r, [{'a': a, 'b': 1} for a in (-3, 0, 4, 126)])
    scopes = {v['scope'] for v in res.provenance.values()}
    assert any(s.startswith('module:') for s in scopes), scopes
    assert 'def _m0' in res.source and name in res.source


def test_map_and_filter_over_a_registered_module():
    r = Registry()
    pred, _ = build((('u', I8),), [('p', 'gt', ('u', 'zero'), None, None)], (('y', 'p'),),
                    constants=(('zero', Value.of(I8, 0)),), registry=r)
    pname = r.register_module(pred)
    doubler, _ = build((('u', I8),), [('d', 'mul', ('u', 'two'), None, None)], (('y', 'd'),),
                       constants=(('two', Value.of(I8, 2)),), registry=r)
    dname = r.register_module(doubler)
    S = setof(I8, 4)
    outer, _ = build((('s', S),),
                     [('k', 'filter', ('s',), {'module': pname}, None),
                      ('m', 'map', ('k',), {'module': dname}, None)],
                     (('k', 'k'), ('m', 'm')), registry=r)
    agree(outer, r, [{'s': (-2, 1, 3)}, {'s': ()}, {'s': (60, 70)}])


def test_constant_folding_and_dead_code_elimination():
    consts = (('one', Value.of(I8, 1)), ('two', Value.of(I8, 2)))
    spec = [('k', 'add', ('one', 'two'), None, None),         # foldable
            ('dead', 'mul', ('a', 'a'), None, None),          # unreachable
            ('live', 'add', ('a', 'k'), None, None)]
    p, r = build((('a', I8),), spec, (('y', 'live'),), constants=consts)
    # the compiler compiles `program.pruned()`, which is what `save_program`
    # already ships; `dead` overflows at a = 60 and is gone from both
    res, mod = agree(p.pruned(), r, [{'a': a} for a in (-5, 0, 60)])
    assert res.stats['nodes_folded'] >= 1
    assert res.stats['nodes_emitted'] == 1, res.stats
    assert '# run/dead' not in res.source


def test_generated_source_is_standalone_and_auditable():
    p, r = build((('a', I8), ('b', I8)), [('s', 'add', ('a', 'b'), None, None)], (('s', 's'),))
    res = compile_program(p, r)
    body = res.source.split('"""', 2)[2]
    assert 'import tcn' not in body and 'import torch' not in body
    assert set(l.split()[1] for l in body.splitlines() if l.startswith('import ')) <= {
        'math', 'cmath', 'struct'}
    assert 'run/s := add(a, b)' in res.source
    prov = [v for v in res.provenance.values() if v.get('operator') == 'add']
    assert prov and prov[0]['node'] == 's' and prov[0]['sources'] == ['a', 'b']
    assert res.types and isinstance(res.types[0], dict)


def test_boundary_still_validates_input_values():
    p, r = build((('a', I8),), [('s', 'add', ('a', 'a'), None, None)], (('s', 's'),))
    mod = compile_program(p, r).module()
    with pytest.raises(OverflowError):
        mod.run({'a': 200})
    with pytest.raises(ValueError):
        mod.run({'a': 1.5})
    with pytest.raises(TypeError):
        mod.run({'a': True})


def test_compilation_requires_a_frozen_program():
    op_a = R.resolve('add', (I8, I8))
    op_b = R.resolve('sub', (I8, I8))
    node = Node('s', I8, (Candidate(op_a, ('a', 'b')), Candidate(op_b, ('a', 'b'))), 'core', 1)
    p = Program((('a', I8), ('b', I8)), (node,), (('s', 's'),)).validate(R)
    with pytest.raises(ValueError):
        compile_program(p, R)


# ---------------------------------------------------------------------------
# Typed-guard elimination -- `research/emitter-guards/RESULTS.md`.
#
# A guard is removed only where the declared types prove it unreachable.  Each
# class is tested twice: once where the proof goes through (the value must be
# identical to the interpreter's and the guard must actually be gone) and once
# where it does not (the guard must still be there and must raise the identical
# exception at the identical edge).  `research/residual-gap/RESULTS.md` sec 2.1
# is why the second half exists: a transform that looked obviously safe was
# wrong on 243 of 2,883 records.
# ---------------------------------------------------------------------------
U8B = integer(8, signed=False, bounds=(0, 3))
U8D = integer(8, signed=False, bounds=(1, 9))
U8W = integer(8, signed=False, bounds=(0, 4))
U16 = integer(16, signed=False)


def _guards(program, registry):
    res = compile_program(program, registry)
    return res, {k[6:]: v for k, v in res.stats.items() if k.startswith('guard_')}


def test_g1_range_guard_goes_where_the_type_proves_it_and_stays_where_it_does_not():
    # min / max of two values of one type cannot leave that type's range
    p, r = build((('a', I8), ('b', I8)),
                 [('m', 'min', ('a', 'b'), None, None), ('x', 'max', ('a', 'b'), None, None)],
                 (('m', 'm'), ('x', 'x')))
    res, g = _guards(p, r)
    assert g['range_eliminated'] == 2 and g['range_emitted'] == 0
    assert '_ovf(' not in res.source
    agree(p, r, [{'a': a, 'b': b} for a in (-128, -1, 0, 127) for b in (-128, 0, 127)])

    # add of two full-width values can overflow, so the guard has to stay
    p, r = build((('a', I8), ('b', I8)), [('s', 'add', ('a', 'b'), None, None)], (('s', 's'),))
    res, g = _guards(p, r)
    assert g['range_eliminated'] == 0 and g['range_emitted'] == 1
    assert '_ovf(' in res.source
    agree(p, r, [{'a': 127, 'b': 1}, {'a': -128, 'b': -1}, {'a': 100, 'b': -100}])


def test_g1_range_guard_goes_for_a_boolean_conversion():
    p, r = build((('a', BOOL),), [('e', 'encode', ('a',), None, U16)], (('e', 'e'),))
    res, g = _guards(p, r)
    assert g['range_eliminated'] == 1 and g['range_emitted'] == 0
    assert '_ovf(' not in res.source
    agree(p, r, [{'a': True}, {'a': False}])


def test_g2_index_guard_goes_only_when_the_bound_is_inside_the_tuple():
    T4 = product(U8, U8, U8, U8)
    # bounds (0, 3) on a 4-tuple: provable
    p, r = build((('t', T4), ('i', U8B)), [('v', 'index', ('t', 'i'), None, None)], (('v', 'v'),))
    res, g = _guards(p, r)
    assert g['index_eliminated'] == 1 and g['index_emitted'] == 0
    assert 'index outside tuple' not in res.source
    agree(p, r, [{'t': (7, 8, 9, 10), 'i': i} for i in range(4)])

    # bounds (0, 4) on a 4-tuple: a closed refinement bound never proves a
    # half-open index, so the guard stays -- and it fires at i = 4
    p, r = build((('t', T4), ('i', U8W)), [('v', 'index', ('t', 'i'), None, None)], (('v', 'v'),))
    res, g = _guards(p, r)
    assert g['index_eliminated'] == 0 and g['index_emitted'] == 1
    assert 'index outside tuple' in res.source
    agree(p, r, [{'t': (7, 8, 9, 10), 'i': i} for i in range(5)])
    with pytest.raises(IndexError):
        compile_program(p, r).module().run({'t': (7, 8, 9, 10), 'i': 4})


def test_g3_zero_denominator_guard_goes_only_when_zero_is_outside_the_bound():
    p, r = build((('a', U8D), ('b', U8D)),
                 [('q', 'idiv', ('a', 'b'), None, None), ('m', 'mod', ('a', 'b'), None, None)],
                 (('q', 'q'), ('m', 'm')))
    res, g = _guards(p, r)
    assert g['zerodiv_eliminated'] == 2 and g['zerodiv_emitted'] == 0
    assert 'zero denominator' not in res.source
    agree(p, r, [{'a': a, 'b': b} for a in (1, 7, 9) for b in (1, 3, 9)])

    p, r = build((('a', U8), ('b', U8)),
                 [('q', 'idiv', ('a', 'b'), None, None)], (('q', 'q'),))
    res, g = _guards(p, r)
    assert g['zerodiv_eliminated'] == 0 and g['zerodiv_emitted'] == 1
    assert 'zero denominator' in res.source
    agree(p, r, [{'a': 7, 'b': 0}, {'a': 7, 'b': 2}])


def test_g4_shift_guard_goes_only_when_the_count_is_bounded_by_the_width():
    p, r = build((('a', U8B), ('b', U8B)), [('s', 'shl', ('a', 'b'), None, None)], (('s', 's'),))
    res, g = _guards(p, r)
    assert g['shift_eliminated'] == 1 and g['shift_emitted'] == 0
    assert 'shift outside bit width' not in res.source
    agree(p, r, [{'a': a, 'b': b} for a in range(4) for b in range(4)])

    p, r = build((('a', I8), ('b', I8)), [('s', 'shl', ('a', 'b'), None, None)], (('s', 's'),))
    res, g = _guards(p, r)
    assert g['shift_eliminated'] == 0 and g['shift_emitted'] == 1
    assert 'shift outside bit width' in res.source
    agree(p, r, [{'a': 1, 'b': b} for b in (0, 3, 8, -1)])


def test_a_load_bearing_clamp_is_never_deleted_and_is_what_proves_the_index():
    """The `research/residual-gap` R5a case, as a standing regression.

    R5a failed on 243 of 2,883 records because it deleted a `min(., 3069)`
    clamp along with the guards.  This pass deletes guards only: the clamp is an
    operator, so it survives, and the interval it establishes is precisely what
    discharges the index obligation downstream.
    """
    T8 = product(*([U8] * 8))
    p, r = build((('t', T8), ('a', U16)),
                 [('c', 'min', ('a', 'k'), None, None),
                  ('v', 'index', ('t', 'c'), None, None)],
                 (('v', 'v'),), constants=(('k', Value.of(U16, 7)),))
    res, g = _guards(p, r)
    assert 'min(' in res.source                        # the clamp survives
    assert g['index_eliminated'] == 1 and g['index_emitted'] == 0
    assert 'index outside tuple' not in res.source
    agree(p, r, [{'t': tuple(range(10, 18)), 'a': a} for a in (0, 3, 7, 8, 4095, 65535)])


def test_guard_elimination_never_changes_the_boundary():
    """Every input is still validated in full; only internal checks are removed."""
    p, r = build((('a', I8), ('b', I8)), [('m', 'min', ('a', 'b'), None, None)], (('m', 'm'),))
    mod = compile_program(p, r).module()
    with pytest.raises(OverflowError):
        mod.run({'a': 200, 'b': 0})
    with pytest.raises(ValueError):
        mod.run({'a': 1.5, 'b': 0})
    with pytest.raises(TypeError):
        mod.run({'a': True, 'b': 0})
