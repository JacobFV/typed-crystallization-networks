"""The `logic` generator's gate-relation observation channel.

`program` is a tuple of `3 * depth` scalars, so its *type* changes with depth and
a fixed-width typed program cannot accept an episode of unseen depth. The gate
relation is the same state under the typed view ARCHITECTURE.md section 1 already
names for sequences and relations, and its type does not depend on depth.
"""
import pytest

from tcn.generation import Host
from tcn.types import Value
from generators.logic.generator import (gate_set_type, gate_set_value, gates_from_set,
                                        GATE_ELEMENT, Implementation)

BASE = {'inputs': 4, 'nondegenerate': True}


def episode(depth, **extra):
    return Host.create('logic', seed=0, index=7, split='test',
                       configuration=BASE | {'depth': depth} | extra)


def test_channel_absent_by_default():
    """An addition, not a change: the default observation set is untouched."""
    view = episode(3).view().observations
    assert set(view) == {'bits', 'program', 'goal'}
    assert 'gate_capacity' not in episode(3).state


def test_channel_present_when_configured():
    view = episode(3, gate_capacity=8).view().observations
    assert set(view) == {'bits', 'program', 'goal', 'gates'}


def test_type_is_independent_of_depth_while_program_is_not():
    types, widths = set(), set()
    for depth in (1, 2, 3, 4, 6, 8):
        view = episode(depth, gate_capacity=8).view().observations
        types.add(view['gates'].type)
        widths.add(len(view['program'].type.items))
        assert len(view['gates'].raw) == depth
    assert len(types) == 1
    assert widths == {3, 6, 9, 12, 18, 24}


def test_round_trip_is_lossless():
    for depth in (1, 2, 5, 8):
        host = episode(depth, gate_capacity=8)
        assert gates_from_set(host.view().observations['gates']) == host.state['gates']


def test_index_field_keeps_the_order_a_set_would_lose():
    """Two gates can share wiring and table; the index field keeps them distinct."""
    gates = [[0, 1, 6], [0, 1, 6], [4, 5, 6]]
    value = gate_set_value(gates, 8)
    assert len(value.raw) == 3
    assert gates_from_set(value) == gates


def test_capacity_is_declared_and_enforced():
    with pytest.raises(ValueError, match='exceeds gate_capacity'):
        episode(9, gate_capacity=8)
    with pytest.raises(ValueError, match='gate_capacity must be'):
        episode(2, gate_capacity=0)
    with pytest.raises(OverflowError):
        gate_set_value([[0, 1, 6]] * 3, 2)


def test_element_semantics_match_the_generators_own_evaluator():
    """(i, a, b, t) means wire w+i = t(wire a, wire b); wire j < w is input bit j."""
    width = 4
    host = episode(6, gate_capacity=8, inputs=width)
    gates = gates_from_set(host.view().observations['gates'])
    values = list(host.state['bits'])
    for a, b, table in gates:
        values.append(bool((table >> (2 * int(values[a]) + int(values[b]))) & 1))
    assert values == host.state['values']
    assert values[width + len(gates) - 1] == host.state['values'][-1]


def test_wire_index_fits_the_declared_field():
    """Capacity bound: the largest wire index is inputs + capacity - 1, and the
    declared int[8] carrier holds 0..255. The generator's own limits (inputs <= 16,
    depth <= 64) keep every configuration inside it, and the field refuses
    anything that would not fit rather than wrapping."""
    view = Host.create('logic', seed=0, index=0, split='test',
                       configuration={'inputs': 16, 'depth': 64, 'gate_capacity': 64}
                       ).view().observations['gates']
    assert max(max(row) for row in view.decoded) < 2 ** 8
    with pytest.raises(OverflowError):
        gate_set_value([[300, 0, 6]], 4)


def test_type_helpers_agree():
    assert gate_set_type(8).items[0] == GATE_ELEMENT
    assert gate_set_type(8).capacity == 8
    assert isinstance(gate_set_value([[0, 1, 6]], 4), Value)


def test_carries_exactly_the_information_program_already_carried():
    """Not a new privilege: the same fields the `program` channel already exposed,
    under a type whose width does not depend on depth."""
    for depth in (1, 2, 3, 8):
        view = episode(depth, gate_capacity=8).view().observations
        flat = [int(x) for x in view['program'].decoded]
        assert gates_from_set(view['gates']) == [flat[i:i + 3] for i in range(0, len(flat), 3)]
