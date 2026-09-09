"""Seasonal crystallization: reversible commitment, pruning, and termination.

`seasons=0` is the default, so every test here contrasts against the monotone
schedule the scheduler ships. The invariants that matter are that a release is
a real inverse of a freeze (the choice trains again, the value path is
differentiable again), that a declared selection is never released, that the
cycle provably terminates, and that pruning only ever commits nodes no live node
reads.
"""
import torch
from tcn.types import BOOL
from tcn.graph import Program,Node,Candidate
from tcn.operators import Registry
from tcn.learning import SoftProgram
from tcn.crystallize import Crystallizer,Objective

r=Registry()
INPUTS={'a':torch.tensor([1.]),'b':torch.tensor([0.])}

def two_choice_program():
    z=Node('z',BOOL,tuple(Candidate(r.resolve(name,(BOOL,BOOL)),('a','b')) for name in ['and','or']))
    w=Node('w',BOOL,tuple(Candidate(r.resolve(name,(BOOL,)),('z',)) for name in ['identity','not']),'readout',2)
    return Program((('a',BOOL),('b',BOOL)),(z,w),(('out','w'),)).validate(r)

def dead_branch_program():
    """`spur` is read by nothing once `w` commits to reading `z`."""
    z=Node('z',BOOL,tuple(Candidate(r.resolve(name,(BOOL,BOOL)),('a','b')) for name in ['and','or']))
    spur=Node('spur',BOOL,tuple(Candidate(r.resolve(name,(BOOL,BOOL)),('a','b')) for name in ['xor','nand']))
    w=Node('w',BOOL,(Candidate(r.resolve('identity',(BOOL,)),('z',)),
                     Candidate(r.resolve('identity',(BOOL,)),('spur',))),'readout',2)
    return Program((('a',BOOL),('b',BOOL)),(z,spur,w),(('out','w'),)).validate(r)

def task_loss(model,port='out',target=1.):
    def loss():
        out,_=model(INPUTS)
        return (out[port]-target).square().mean()
    return loss

def test_thaw_is_the_inverse_of_freeze():
    m=SoftProgram(two_choice_program(),r)
    m.freeze('z',0)
    assert 'z' in m.frozen and not m.choices[0].requires_grad
    m.thaw('z')
    assert 'z' not in m.frozen and 'z' not in m.pinned and m.choices[0].requires_grad
    # The value path is differentiable again: a frozen node executes exactly and
    # detached, so before the release the downstream loss reaches no logit at all.
    loss=task_loss(m)()
    grads=torch.autograd.grad(loss,[m.choices[0],m.choices[1]],allow_unused=True)
    assert all(g is not None for g in grads)

def test_thaw_refuses_a_declared_selection():
    from dataclasses import replace
    program=two_choice_program()
    declared=replace(program,nodes=(replace(program.nodes[0],selected=0),program.nodes[1])).validate(r)
    m=SoftProgram(declared,r)
    try:
        m.thaw('z')
    except ValueError as e:
        assert 'prior' in str(e)
    else:
        raise AssertionError('a declared selection must not be released')
    # And a node that was never frozen is not releasable either.
    try:
        m.thaw('w')
    except ValueError as e:
        assert 'not frozen' in str(e)
    else:
        raise AssertionError('an unfrozen node must not be released')

def test_seasons_release_and_recommit_and_the_run_still_ends_frozen():
    m=SoftProgram(two_choice_program(),r);opt=torch.optim.Adam(m.parameters(),lr=.1)
    s=Crystallizer(m,opt,seasons=2,season_rounds=3,thaw_fraction=1.,thaw_limit=1,summer_steps=5)
    s.run(Objective.of(task_loss(m)),rounds=6,retrain_steps=3)
    assert s.season_log, 'at least one summer must have run'
    assert all(len(e.thawed) for e in s.season_log)
    assert len(m.frozen)==len(m.program.nodes), 'the run must end in a winter, fully frozen'
    # The frozen count is not monotone: every summer lowers it.
    assert any(e.frozen_after<e.frozen_before for e in s.season_log)

def test_the_cycle_terminates_by_the_thaw_limit():
    m=SoftProgram(two_choice_program(),r);opt=torch.optim.Adam(m.parameters(),lr=.1)
    s=Crystallizer(m,opt,seasons=50,season_rounds=2,thaw_fraction=1.,thaw_decay=1.,thaw_limit=1,summer_steps=2)
    s.run(Objective.of(task_loss(m)),rounds=4,retrain_steps=2)
    # 50 summers were allowed; the per-node release limit stops it far sooner,
    # and no node is ever released more than once.
    assert len(s.season_log)<50
    assert all(v<=1 for v in s.thaw_counts.values())

def test_thaw_never_releases_a_settled_node():
    m=SoftProgram(two_choice_program(),r);opt=torch.optim.Adam(m.parameters(),lr=.1)
    s=Crystallizer(m,opt,seasons=3,season_rounds=3,thaw_fraction=1.,thaw_limit=9,summer_steps=3)
    s.run(Objective.of(task_loss(m)),rounds=6,retrain_steps=3)
    # Settled is only assigned after a winter confirms the same candidate, so the
    # invariant to check is forward-looking: no settled node is ever offered again.
    assert all(n not in s.thaw_candidates(1.) for n in s.settled)

def test_pruning_only_commits_nodes_no_live_node_reads():
    m=SoftProgram(dead_branch_program(),r);opt=torch.optim.Adam(m.parameters(),lr=.1)
    s=Crystallizer(m,opt,prune=True)
    assert s.live_nodes()>={'z','spur','w'}, 'while w is soft both branches are live'
    m.freeze('w',0)                       # w commits to reading z
    assert 'spur' not in s.live_nodes()
    assert s.prune_dead()==['spur']
    assert 'spur' in m.frozen and 'spur' in s.pruned
    # A pruned node is dead, so it is never a release candidate.
    assert 'spur' not in s.thaw_candidates(1.)

def test_the_closing_block_trial_is_transactional_and_off_by_default():
    m=SoftProgram(two_choice_program(),r);opt=torch.optim.Adam(m.parameters(),lr=.1)
    s=Crystallizer(m,opt)
    assert s.close_block is False
    # A block trial that cannot reach the target must roll the whole block back.
    unreachable=Objective.of(lambda: torch.tensor(float('nan'),requires_grad=True))
    event=s.try_freeze_block(['z','w'],unreachable,retrain_steps=0)
    assert not event.accepted and not m.frozen
    # An achievable one commits every named node at once.
    s2=Crystallizer(m,opt,tolerance=10.)
    event=s2.try_freeze_block(['z','w'],Objective.of(task_loss(m)),retrain_steps=1)
    assert event.accepted and set(m.frozen)=={'z','w'}

def test_seasons_are_off_by_default():
    m=SoftProgram(two_choice_program(),r);opt=torch.optim.Adam(m.parameters(),lr=.1)
    s=Crystallizer(m,opt)
    assert s.seasons==0 and s.prune is False and s.close_block is False
    s.run(Objective.of(task_loss(m)),rounds=4,retrain_steps=2)
    assert s.season_log==[] and s.thaw_counts=={} and s.pruned==[]
