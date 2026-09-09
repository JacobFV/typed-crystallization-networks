"""The compiled external boundary's identity guard: sound, or it does not ship.

`research/program-length/RESULTS.md` Q2.  A homogeneous wide tuple's boundary
short-circuits to `return x` when the container is provably already canonical,
which is the whole of the 11x measured there.  The guard is a *recogniser*: it
may only accept a value the full boundary would have returned unchanged, and
anything it rejects falls through to the unmodified path, so the exception type,
its message and the element it is raised at are all preserved.

The hazard the guard has to survive is that `hash(True) == hash(1.0) == hash(1)`,
so a bare membership or interval test accepts a `bool` or an admissible-valued
`float` and would hand the program a non-canonical carrier where the boundary is
required to raise `TypeError`.  Every test below exists to hold that line.
"""
import pytest

from tcn.compile import compile_program
from tcn.graph import Candidate, Node, Program
from tcn.operators import Registry
from tcn.types import Value, fixed, floating, integer, product

U8 = integer(8, signed=False)
POS = integer(32, signed=False, bounds=(0, 4096))


def _passthrough(element, arity, registry=None):
    """A one-node program whose only input is a homogeneous tuple of `element`."""
    r = registry or Registry()
    T = product(*(element for _ in range(arity)))
    op = r.resolve('project', (T,), element, {'index': 0})
    p = Program((('a', T),), (Node('s', op.output, (Candidate(op, ('a',)),), 'core', 1, 0),),
                (('s', 's'),)).validate(r)
    return T, compile_program(p, r), r


def test_guard_is_emitted_for_a_wide_integer_tuple_and_returns_by_reference():
    T, res, _ = _passthrough(U8, 16)
    assert res.stats['boundary_identity_guards'] == 1
    mod = res.module()
    good = tuple(range(16))
    assert mod._IN['a'](good) is good, 'a canonical container must not be rebuilt'
    assert mod.run({'a': good})[0]['s'] == 0


@pytest.mark.parametrize('odd', [True, False, 1.0, 0.0, 15.0, 0.5, 256, -1, 'x', None,
                                 10 ** 400, float('nan'), float('inf')])
def test_guard_matches_the_interpreter_exactly_on_every_non_canonical_element(odd):
    """Same value or same exception, whichever the interpreter produces.

    Note the two distinct outcomes, both of which the guard has to get right: a
    `bool` is *rejected* by `Value.of` with a `TypeError`, while an integral
    `float` such as `1.0` is *accepted* and canonicalised to the `int` `1`.  The
    guard must therefore decline the identity branch for both -- returning the
    container unchanged would be wrong in the second case just as much as in the
    first, and for a reason the type graph knows: `hash(1.0) == hash(1)`.
    """
    T, res, _ = _passthrough(U8, 16)
    mod = res.module()
    broken = (odd,) + tuple(range(1, 16))

    def outcome(fn):
        try:
            return ('ok', fn())
        except Exception as e:                       # noqa: BLE001
            return ('raise', type(e).__name__, str(e))

    oracle = outcome(lambda: Value.of(T, broken).decoded)
    generated = outcome(lambda: mod._IN['a'](broken))
    assert generated == oracle, (odd, generated, oracle)
    if oracle[0] == 'ok':
        assert generated[1] is not broken, (odd, 'a non-canonical container was passed through')


def test_guard_respects_refinement_bounds_narrower_than_the_encoded_width():
    """`POS` is 32-bit but bounded to (0, 4096); 4097 must still raise."""
    T, res, _ = _passthrough(POS, 8)
    assert res.stats['boundary_identity_guards'] == 1
    mod = res.module()
    assert mod._IN['a']((0, 1, 2, 3, 4, 5, 6, 4096)) == (0, 1, 2, 3, 4, 5, 6, 4096)
    with pytest.raises(ValueError):
        mod._IN['a']((0, 1, 2, 3, 4, 5, 6, 4097))
    with pytest.raises(ValueError):
        Value.of(T, (0, 1, 2, 3, 4, 5, 6, 4097))


@pytest.mark.parametrize('element', [floating(32), floating(64), fixed(16, scale=4)])
def test_no_guard_where_the_round_trip_is_not_the_identity(element):
    _, res, _ = _passthrough(element, 16)
    assert res.stats['boundary_identity_guards'] == 0


def test_no_guard_on_a_narrow_tuple_which_never_took_the_map_path():
    _, res, _ = _passthrough(U8, 3)
    assert res.stats['boundary_identity_guards'] == 0


def test_wide_interval_falls_back_to_min_max_and_stays_exact():
    """Above `_IDENTITY_SET_MAX` the guard uses `min`/`max`, which is unbounded."""
    wide = integer(32, signed=True)
    T, res, _ = _passthrough(wide, 16)
    assert res.stats['boundary_identity_guards'] == 1
    assert 'min(x)' in res.source and 'max(x)' in res.source
    mod = res.module()
    good = tuple(range(-8, 8))
    assert mod._IN['a'](good) is good
    with pytest.raises(TypeError):
        mod._IN['a']((True,) + good[1:])
    with pytest.raises(OverflowError):
        mod._IN['a']((1 << 40,) + good[1:])
