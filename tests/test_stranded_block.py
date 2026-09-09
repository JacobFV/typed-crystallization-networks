"""The per-node viability guard is incomplete over sets, and the block trial closes it.

The guard rejects a freeze whose remaining trainable region is severed from the
task objective. Correct per node, incomplete over sets: a residual set can reach
a state where freezing any single member disconnects the others, so every
single-node trial is refused forever and the run never closes. Measured on
`joint` at 40 episodes with the shipped crystallizer and no seasons machinery,
1 of 8 seeds ends 9/13 frozen behind 29 such refusals
(`research/stranded-block/RESULTS.md`).

What the block trial must not become is an escape hatch. It carries the same
tolerance, the same conformance check and the same all-or-nothing rollback, and
these tests pin that down.
"""
import torch

from tcn.types import BOOL
from tcn.graph import Program, Node, Candidate
from tcn.operators import Registry
from tcn.learning import SoftProgram
from tcn.crystallize import Crystallizer, Objective

r = Registry()
INPUTS = {'a': torch.tensor([1.]), 'b': torch.tensor([0.])}


def two_choice_program():
    z = Node('z', BOOL, tuple(Candidate(r.resolve(name, (BOOL, BOOL)), ('a', 'b'))
                              for name in ['and', 'or']))
    w = Node('w', BOOL, tuple(Candidate(r.resolve(name, (BOOL,)), ('z',))
                              for name in ['identity', 'not']), 'readout', 2)
    return Program((('a', BOOL), ('b', BOOL)), (z, w), (('out', 'w'),)).validate(r)


def task_loss(model, port='out', target=1.):
    def loss():
        out, _ = model(INPUTS)
        return (out[port] - target).square().mean()
    return loss


def test_a_block_that_cannot_reach_the_target_rolls_back_whole():
    m = SoftProgram(two_choice_program(), r)
    s = Crystallizer(m, torch.optim.Adam(m.parameters(), lr=.1))
    unreachable = Objective.of(lambda: torch.tensor(float('nan'), requires_grad=True))
    event = s.try_freeze_block(['z', 'w'], unreachable, retrain_steps=0)
    assert not event.accepted
    assert not m.frozen, "a refused block must leave no member frozen"


def test_an_achievable_block_commits_every_member_at_once():
    m = SoftProgram(two_choice_program(), r)
    s = Crystallizer(m, torch.optim.Adam(m.parameters(), lr=.1), tolerance=10.)
    event = s.try_freeze_block(['z', 'w'], Objective.of(task_loss(m)), retrain_steps=1)
    assert event.accepted and set(m.frozen) == {'z', 'w'}
    assert s.block_events == [event]


def test_the_block_trial_obeys_the_degradation_tolerance():
    """It is a completeness fix for the guard, not a bypass of the freeze contract."""
    m = SoftProgram(two_choice_program(), r)
    s = Crystallizer(m, torch.optim.Adam(m.parameters(), lr=.1), tolerance=-1e9)
    event = s.try_freeze_block(['z', 'w'], Objective.of(task_loss(m)), retrain_steps=0)
    assert not event.accepted and event.reason == "block degradation"
    assert not m.frozen


def test_the_block_trial_obeys_conformance():
    m = SoftProgram(two_choice_program(), r)
    s = Crystallizer(m, torch.optim.Adam(m.parameters(), lr=.1), tolerance=10.)
    event = s.try_freeze_block(['z', 'w'], Objective.of(task_loss(m)),
                               retrain_steps=1, conformance=lambda program: False)
    assert not event.accepted and event.reason == "block runtime conformance"
    assert not m.frozen


def test_a_run_that_already_closes_never_reaches_the_block_trial():
    """The fix must be inert wherever the per-node guard was already sufficient."""
    m = SoftProgram(two_choice_program(), r)
    s = Crystallizer(m, torch.optim.Adam(m.parameters(), lr=.1), tolerance=10.)
    s.run(Objective.of(task_loss(m)), rounds=6, retrain_steps=2)
    assert len(m.frozen) == len(m.program.nodes)
    assert s.block_events == [], "a program the rounds closed needs no block trial"


def test_the_block_trial_can_be_switched_off():
    m = SoftProgram(two_choice_program(), r)
    s = Crystallizer(m, torch.optim.Adam(m.parameters(), lr=.1), close_block=False)
    assert s.close_block is False
    s.run(Objective.of(task_loss(m)), rounds=1, retrain_steps=1)
    assert s.block_events == []


def test_a_deliberately_deferred_run_gets_no_block_trial():
    """The trial fires on evidence of the guard's incompleteness, not on any
    unfinished run. An eligibility gate that declines to commit must not have a
    commitment forced on it at the end."""
    m = SoftProgram(two_choice_program(), r)
    s = Crystallizer(m, torch.optim.Adam(m.parameters(), lr=.1),
                     eligibility="plateau", plateau_patience=10 ** 6)
    s.run(Objective.of(task_loss(m)), rounds=3, retrain_steps=1)
    assert s.events == [] and s.block_events == []
    assert m.frozen == {}, "the gate deferred every freeze; nothing may be frozen"
