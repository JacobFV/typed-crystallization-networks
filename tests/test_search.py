"""The discrete reference: enumeration over the same candidate space."""
import pytest
from examples.mixed import problem
from tcn.search import enumerate_fit, space_size, evaluate
from tcn.operators import Registry
from tcn.types import BOOL, Value, floating
from tcn.graph import Program, Node, Candidate, Signal

def test_enumeration_solves_the_mixed_scaffold_and_certifies_uniqueness():
    p, signals, examples = problem()
    assert space_size(p) == 96
    found = enumerate_fit(p, examples, signals, tolerance=.005)
    assert found.solved and found.exhausted
    assert found.unique is True
    assert found.evaluated == found.space_size
    assert found.selections['logic'] == 6          # xor over (a, b)

def test_stop_at_first_forfeits_the_uniqueness_certificate():
    p, signals, examples = problem()
    found = enumerate_fit(p, examples, signals, tolerance=.005, stop_at_first=True)
    assert found.solved and found.unique is None
    assert found.evaluated < found.space_size

def test_budget_limits_the_search_and_reports_it_did_not_exhaust():
    p, signals, examples = problem()
    found = enumerate_fit(p, examples, signals, tolerance=.005, max_programs=4)
    assert found.evaluated == 4 and not found.exhausted and found.unique is None

def test_trainable_constants_are_reported_as_outside_the_discrete_search():
    r = Registry(); F = floating()
    nodes = (Node('y', F, (Candidate(r.resolve('mul', (F, F)), ('x', 'k')),), 'core', 1),)
    p = Program((('x', F),), nodes, (('out', 'y'),), (('k', Value.of(F, 2.)),), trainable_constants=('k',))
    signals = (Signal('y', 'y', ('core',), F),)
    examples = [{'inputs': {'x': Value.of(F, 1.)}, 'targets': {'y': Value.of(F, 2.)}}]
    found = enumerate_fit(p, examples, signals, registry=r)
    assert found.continuous == ('k',)

def test_an_illegal_numeric_domain_is_unusable_rather_than_an_error():
    r = Registry(); F = floating()
    nodes = (Node('y', F, (Candidate(r.resolve('log', (F,)), ('x',)),), 'core', 1),)
    p = Program((('x', F),), nodes, (('out', 'y'),))
    signals = (Signal('y', 'y', ('core',), F),)
    bad = [{'inputs': {'x': Value.of(F, -1.)}, 'targets': {'y': Value.of(F, 0.)}}]
    assert evaluate(p, {'y': 0}, bad, signals, r) is None
    found = enumerate_fit(p, bad, signals, registry=r)
    assert not found.solved and found.exhausted
