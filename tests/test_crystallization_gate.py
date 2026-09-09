"""The loss-gated commitment schedule.

`eligibility="plateau"` makes a node eligible to freeze only once the
unregularized task loss has stopped improving, and `anneal="plateau"` puts the
temperature/quantization schedule on the same signal. Both default off, so the
fixed-clock behaviour these tests contrast against is what the scheduler ships.
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
    """Two independent real decisions, so `run` has more than one commitment to make."""
    z=Node('z',BOOL,tuple(Candidate(r.resolve(name,(BOOL,BOOL)),('a','b')) for name in ['and','or']))
    w=Node('w',BOOL,tuple(Candidate(r.resolve(name,(BOOL,)),('z',)) for name in ['identity','not']),'readout',2)
    return Program((('a',BOOL),('b',BOOL)),(z,w),(('out','w'),)).validate(r)

def task_loss(model,port,target=1.):
    def loss():
        out,_=model(INPUTS)
        return (out[port]-target).square().mean()
    return loss

def test_plateau_detector_reads_relative_improvement_over_its_window():
    m=SoftProgram(two_choice_program(),r);opt=torch.optim.Adam(m.parameters(),lr=.1)
    s=Crystallizer(m,opt,plateau_window=3,plateau_tolerance=1e-3)
    # Fewer than window+1 readings: nothing to compare against yet.
    s.progress=[1.,.9,.8];assert not s.plateaued()
    # Still falling by 12.5% against the reading three rounds back.
    s.progress=[1.,.9,.8,.7];assert not s.plateaued()
    # Flat, and a rising objective, both count as "stopped improving".
    s.progress=[1.,1.,1.,1.];assert s.plateaued()
    s.progress=[1.,1.1,1.2,1.3];assert s.plateaued()
    # Improvement below the relative threshold counts as stopped.
    s.progress=[1.,1.,1.,1.-5e-4];assert s.plateaued()
    s.progress=[1.,1.,1.,1.-5e-2];assert not s.plateaued()
    # An unusable prior reading disarms rather than opening the gate.
    s.progress=[float('inf'),1.,1.,1.];assert not s.plateaued()

def test_plateau_eligibility_defers_the_first_freeze_and_trains_instead():
    m=SoftProgram(two_choice_program(),r);opt=torch.optim.Adam(m.parameters(),lr=.1)
    loss=task_loss(m,'out')
    # The fixed clock commits in round one.
    prompt=Crystallizer(m,opt,tolerance=float('inf'))
    prompt.run(loss,rounds=1,retrain_steps=1)
    assert len(m.frozen)>0 and prompt.gate_evaluations==0
    # The gate cannot open before it has window+1 readings, so with that many
    # rounds nothing is frozen and every round is spent descending the objective.
    m=SoftProgram(two_choice_program(),r);opt=torch.optim.Adam(m.parameters(),lr=.1)
    loss=task_loss(m,'out')
    gated=Crystallizer(m,opt,tolerance=float('inf'),eligibility='plateau',plateau_window=3)
    steps={'n':0};original=opt.step
    opt.step=lambda *a,**k:(steps.__setitem__('n',steps['n']+1),original(*a,**k))[1]
    gated.run(loss,rounds=3,retrain_steps=2)
    assert m.frozen=={} and gated.events==[]
    assert gated.gate_evaluations==3 and steps['n']==6      # 3 closed rounds x 2 steps
    assert [g['open'] for g in gated.gate_log]==[False]*3

def test_plateau_eligibility_accepts_one_freeze_per_opening_then_re_arms():
    m=SoftProgram(two_choice_program(),r);opt=torch.optim.Adam(m.parameters(),lr=0.)
    loss=task_loss(m,'out')
    # lr=0 makes the objective flat, so the gate opens as soon as it has a window.
    gated=Crystallizer(m,opt,tolerance=float('inf'),eligibility='plateau',plateau_window=3)
    gated.run(loss,rounds=8,retrain_steps=1)
    opened=[i for i,g in enumerate(gated.gate_log) if g['open']]
    # First opening once the window is full; the second only after a fresh window,
    # because opening re-arms the detector rather than sliding it.
    assert opened==[3,6]
    assert [g['frozen'] for g in gated.gate_log]==[0,0,0,0,1,1,1]
    assert len(m.frozen)==2 and len(gated.events)==2
    # Re-armed: the round after an opening starts a fresh window.
    assert gated.rounds_waited==0

def test_gated_annealing_holds_the_temperature_while_the_loss_still_falls():
    def prepare():
        m=SoftProgram(two_choice_program(),r)
        return m,torch.optim.Adam(m.parameters(),lr=.1),task_loss(m,'out')
    # The fixed clock decays every unfrozen node once per round regardless.
    m,opt,loss=prepare();before=dict(m.temperatures)
    Crystallizer(m,opt,tolerance=-1.).run(loss,rounds=3,retrain_steps=1)
    assert m.temperatures['z']==before['z']*.8**3
    # On the same signal, a still-improving objective holds it.
    m,opt,loss=prepare();before=dict(m.temperatures)
    gated=Crystallizer(m,opt,tolerance=-1.,anneal='plateau',plateau_window=3)
    gated.run(loss,rounds=3,retrain_steps=1)
    assert m.temperatures['z']==before['z']
    assert gated.gate_evaluations==3 and not any(g['open'] for g in gated.gate_log)
    # Eligibility is untouched by the anneal gate: the trials still ran.
    assert len(gated.events)>0

def test_the_gate_reads_the_task_loss_not_the_regularized_total():
    m=SoftProgram(two_choice_program(),r);opt=torch.optim.Adam(m.parameters(),lr=.1)
    task=task_loss(m,'out');seen={'task':0,'total':0}
    def counted_task():
        seen['task']+=1;return task()
    def counted_total():
        seen['total']+=1;return task()+.01*m.entropy()
    gated=Crystallizer(m,opt,tolerance=float('inf'),eligibility='plateau',plateau_window=3)
    gated.run(Objective(counted_total,counted_task),rounds=3,retrain_steps=1)
    # One unregularized reading per round for the gate; the residual training the
    # closed rounds spend descends the regularized total, once per retrain step.
    assert seen['task']==3 and gated.gate_evaluations==3 and seen['total']==3
    assert len(gated.progress)==3 and all(v==v and v!=float('inf') for v in gated.progress)

def test_patience_opens_the_gate_on_an_objective_that_never_settles():
    m=SoftProgram(two_choice_program(),r);opt=torch.optim.Adam(m.parameters(),lr=0.)
    total=task_loss(m,'out')
    falling=iter([1./2**i for i in range(64)])
    # Differentiable, so the connectivity guard still sees a path; scaled down
    # every round, so the window never reports the objective as settled.
    def task():
        return total()*next(falling)
    gated=Crystallizer(m,opt,tolerance=float('inf'),eligibility='plateau',
                       plateau_window=3,plateau_patience=5)
    gated.run(Objective(total,task),rounds=8,retrain_steps=0)
    # The task loss halves every round, so the window never says "stopped"; the
    # patience valve is what lets crystallization proceed at all.
    assert [g['open'] for g in gated.gate_log][:6]==[False]*5+[True]
    assert len(m.frozen)>0

def test_anneal_never_holds_the_schedule_for_the_whole_run():
    m=SoftProgram(two_choice_program(),r);opt=torch.optim.Adam(m.parameters(),lr=.1)
    loss=task_loss(m,'out');before=dict(m.temperatures)
    # The ablation that separates "concentrate later" from "never concentrate".
    s=Crystallizer(m,opt,tolerance=-1.,anneal='never')
    s.run(loss,rounds=4,retrain_steps=1)
    assert m.temperatures==before and s.gate_evaluations==0 and len(s.events)>0

def test_the_gate_rejects_unknown_rules_and_defaults_to_the_fixed_clock():
    m=SoftProgram(two_choice_program(),r);opt=torch.optim.Adam(m.parameters(),lr=.1)
    s=Crystallizer(m,opt)
    assert s.eligibility=='immediate' and s.anneal=='round'
    for kwargs in ({'eligibility':'loss'},{'anneal':'none'},{'plateau_window':0}):
        try:Crystallizer(m,opt,**kwargs);assert False
        except ValueError:pass
