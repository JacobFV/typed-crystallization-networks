"""Perturbation-based selection and the gradient-connectivity guard.

The guard tests in this file are the positive and negative cases for the
interface change in `tcn.crystallize.Objective`: `grad is None` against the
*regularized* objective tests reachability in the autograd graph, which a
discreteness term guarantees for every unfrozen logit; against the *task*
objective it tests what ARCHITECTURE.md section 5 asks for.
"""
import torch
from tcn.types import BOOL,Value
from tcn.graph import Program,Node,Candidate
from tcn.operators import Registry
from tcn.learning import SoftProgram
from tcn.crystallize import Crystallizer,Objective

r=Registry()
INPUTS={'a':torch.tensor([1.]),'b':torch.tensor([0.])}

def choice_program():
    """One real decision: `and` or `or` over the same two inputs."""
    node=Node('z',BOOL,tuple(Candidate(r.resolve(name,(BOOL,BOOL)),('a','b')) for name in ['and','or']))
    return Program((('a',BOOL),('b',BOOL)),(node,),(('out','z'),)).validate(r)

def interior_program():
    """`hidden` reaches the objective only through `out`; freezing `out` severs it."""
    hidden=Node('hidden',BOOL,tuple(Candidate(r.resolve(f'truth_{i}',(BOOL,BOOL)),('a','b')) for i in range(4)),'latent',1)
    out=Node('out',BOOL,tuple(Candidate(r.resolve(name,(BOOL,)),('hidden',)) for name in ['identity','not']),'readout',2)
    return Program((('a',BOOL),('b',BOOL)),(hidden,out),(('answer','out'),)).validate(r)

def task_loss(model,port,target=1.):
    def loss():
        out,_=model(INPUTS)
        return (out[port]-target).square().mean()
    return loss

def test_perturbation_selects_the_candidate_that_carries_the_objective():
    m=SoftProgram(choice_program(),r);opt=torch.optim.Adam(m.parameters(),lr=.1)
    with torch.no_grad():m.choices[0].copy_(torch.tensor([1.,0.]))
    loss=task_loss(m,'out')
    # The logit magnitude points at `and`, which computes 0 where the target is 1.
    assert m.selections()['z']==0
    assert Crystallizer(m,opt,selection='entropy').entropy_candidates()==[]
    scheduler=Crystallizer(m,opt,tolerance=float('inf'))
    assert scheduler.candidates(loss)==['z']
    # Removing `and` leaves the objective satisfied; removing `or` breaks it, so
    # `or` is the candidate whose removal hurts most.
    assert scheduler.perturbation_scores(0,m.program.nodes[0],loss)==[0.,1.]
    assert scheduler.selected['z']==1
    scheduler.run(loss,rounds=1,retrain_steps=0)
    assert m.frozen=={'z':1}
    out,_=m(INPUTS);assert float(out['out'])==1.

def test_single_candidate_nodes_are_ordered_after_real_decisions():
    p=interior_program()
    plumbing=Node('copy',BOOL,(Candidate(r.resolve('identity',(BOOL,)),('out',)),),'readout',3)
    p=Program(p.inputs,p.nodes+(plumbing,),(('answer','copy'),)).validate(r)
    m=SoftProgram(p,r);opt=torch.optim.Adam(m.parameters(),lr=.1)
    order=Crystallizer(m,opt).candidates(task_loss(m,'answer'))
    assert order[-1]=='copy'

def test_connectivity_guard_catches_a_severed_interior_only_with_the_task_objective():
    def prepare():
        m=SoftProgram(interior_program(),r)
        return m,torch.optim.Adam(m.parameters(),lr=.1),task_loss(m,'answer')
    # `total` carries a discreteness regularizer, exactly as integrated training
    # does. It keeps every unfrozen logit attached to the loss graph.
    m,opt,task=prepare()
    total=lambda:task()+.01*m.entropy()
    event=Crystallizer(m,opt,tolerance=float('inf')).try_freeze('out',total,retrain_steps=0)
    assert event.accepted and event.reason=='validated'
    assert torch.autograd.grad(total(),[m.choices[0]],allow_unused=True)[0] is not None
    # Same freeze, same regularized objective for retraining and degradation, but
    # the viability probe now reads the unregularized task loss. `hidden` has no
    # path to it at all once `out` is frozen, so the freeze is deferred.
    m,opt,task=prepare()
    total=lambda:task()+.01*m.entropy()
    event=Crystallizer(m,opt,tolerance=float('inf')).try_freeze('out',Objective(total,task),retrain_steps=0)
    assert not event.accepted and event.reason=='disconnected remaining region'
    assert 'out' not in m.frozen and m.choices[0].requires_grad

def test_connectivity_guard_ignores_a_merely_concentrated_choice():
    m=SoftProgram(interior_program(),r);opt=torch.optim.Adam(m.parameters(),lr=.1)
    task=task_loss(m,'answer');total=lambda:task()+.01*m.entropy()
    # `out` still carries the objective, but its distribution has concentrated.
    with torch.no_grad():m.choices[1].copy_(torch.tensor([200.,0.]))
    grad=torch.autograd.grad(task(),[m.choices[1]],allow_unused=True)[0]
    # The reverted "all-zero gradient means disconnected" rule would flag this
    # node: the softmax Jacobian vanishes at a one-hot distribution whether or
    # not any task signal reaches it. Reachability separates the two cases.
    assert grad is not None and not torch.any(grad!=0)
    event=Crystallizer(m,opt,tolerance=float('inf')).try_freeze('hidden',Objective(total,task),retrain_steps=0)
    assert event.reason!='disconnected remaining region'
    assert event.accepted and m.frozen['hidden']==m.selections()['hidden']

def test_objective_defaults_to_one_closure_and_rejects_unknown_selection():
    m=SoftProgram(choice_program(),r);opt=torch.optim.Adam(m.parameters(),lr=.1)
    loss=task_loss(m,'out');objective=Objective.of(loss)
    assert objective.total is loss and objective.task is loss
    assert Objective.of(objective) is objective
    try:Crystallizer(m,opt,selection='entropy_stability');assert False
    except ValueError:pass
