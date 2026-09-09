"""Supervised typed synthesis with declared intermediate signals and freeze evidence."""
from dataclasses import asdict
import torch
from .learning import SoftProgram,tensor
from .crystallize import Crystallizer,Objective

def fit(program,examples,signals,steps=300,lr=.05,freeze=True,registry=None,tolerance=.001,polish=200):
    if not examples:raise ValueError('training examples required')
    if polish<0:raise ValueError('polish steps must be non-negative')
    program.validate_signals(signals)
    model=SoftProgram(program,registry);optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    inputs={k:torch.stack([tensor(ex['inputs'][k]) for ex in examples]) for k,_ in program.inputs}
    targets={s.target:torch.stack([tensor(ex['targets'][s.target]) for ex in examples]) for s in signals}
    def loss_fn():
        _,_,trace=model(inputs,return_trace=True)
        return model.probe_loss(trace,targets,signals)
    history=[];torch.set_num_threads(1)
    for step in range(steps):
        optimizer.zero_grad();loss=loss_fn()+.001*(step/max(1,steps))*model.entropy()
        if loss.requires_grad:loss.backward();optimizer.step()
        if step%25==0 or step==steps-1:history.append({'step':step,'loss':float(loss.detach()),'entropy':float(model.entropy().detach())})
    # A trainable constant sees a gradient blurred by the candidate mixture, so it
    # converges far more slowly than the structure does: a program can select the
    # right operators and still miss exact tolerance by orders of magnitude. Hold
    # the selected structure hard and refine the continuous parameters alone, then
    # restore the choices so crystallization proceeds normally. Neither a decayed
    # nor a raised learning rate substitutes for this; both were measured worse.
    if polish and model.constants:
        held=dict(model.selections());requires=[p.requires_grad for p in model.choices]
        model.trials=dict(held)
        for p in model.choices:p.requires_grad_(False)
        refiner=torch.optim.Adam([p for p in model.constants.values()],lr=lr)
        for step in range(polish):
            refiner.zero_grad();refined=loss_fn()
            if refined.requires_grad:refined.backward();refiner.step()
        model.trials={}
        for p,flag in zip(model.choices,requires):p.requires_grad_(flag)
        history.append({'step':steps+polish,'loss':float(loss_fn().detach()),'entropy':float(model.entropy().detach()),'phase':'polish'})
    scheduler=Crystallizer(model,optimizer,tolerance=tolerance)
    def exact_error(exact):
        """Largest absolute disagreement between the exported program and the targets."""
        worst=0.
        for ex in examples:
            _,_,trace=exact.execute(ex['inputs'],registry=model.registry)
            for signal in signals:
                a=torch.tensor(trace[signal.source].flat());b=torch.tensor(ex['targets'][signal.target].flat())
                worst=max(worst,float((a-b).abs().max()))
        return worst
    def conform(exact):return exact_error(exact)<=tolerance
    # The supervised objective here carries no architecture regularizer -- the
    # entropy term is added in the training loop above and not during residual
    # retraining -- so the task probe and the optimized total are the same
    # closure. Stating it explicitly keeps the connectivity guard's contract
    # visible at the call site rather than resting on a default.
    if freeze:scheduler.run(Objective(loss_fn,loss_fn),rounds=24,retrain_steps=10,conformance=conform)
    # The relaxed loss and the exported program's exact error are separate
    # measurements and disagree systematically: a near-one-hot mixture can reach
    # zero soft loss while its argmax is a different program. Report both, and
    # never treat `loss` as evidence that synthesis succeeded.
    error=exact_error(model.export())
    return model,{'training':history,'freeze_events':[asdict(x) for x in scheduler.events],'fully_frozen':len(model.frozen)==len(program.nodes) and all(not p.requires_grad for p in model.constants.values()),'loss':float(loss_fn().detach()),'relaxed_loss':float(loss_fn().detach()),'exact_max_error':error,'exact_conformance':error<=tolerance,'tolerance':tolerance}
