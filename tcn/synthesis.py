"""Supervised typed synthesis with declared intermediate signals and freeze evidence."""
from dataclasses import asdict
import torch
from .learning import SoftProgram,tensor
from .crystallize import Crystallizer

def fit(program,examples,signals,steps=300,lr=.05,freeze=True,registry=None,tolerance=.001):
    if not examples:raise ValueError('training examples required')
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
    scheduler=Crystallizer(model,optimizer,tolerance=tolerance,entropy_limit=.9)
    def conform(exact):
        for ex in examples:
            _,_,trace=exact.execute(ex['inputs'],registry=model.registry)
            for signal in signals:
                a=torch.tensor(trace[signal.source].flat());b=torch.tensor(ex['targets'][signal.target].flat())
                if float((a-b).abs().max())>tolerance:return False
        return True
    if freeze:scheduler.run(loss_fn,rounds=24,retrain_steps=10,conformance=conform)
    return model,{'training':history,'freeze_events':[asdict(x) for x in scheduler.events],'fully_frozen':len(model.frozen)==len(program.nodes) and all(not p.requires_grad for p in model.constants.values()),'loss':float(loss_fn().detach()),'exact_conformance':conform(model.export())}
