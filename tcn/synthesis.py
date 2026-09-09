"""Supervised typed synthesis with declared intermediate signals and freeze evidence."""
from dataclasses import asdict
import math
import torch
from .learning import SoftProgram,tensor
from .crystallize import Crystallizer,Objective

def _exact_error(program,examples,signals,registry):
    """Largest absolute disagreement between an exact program and the targets."""
    worst=0.
    for ex in examples:
        _,_,trace=program.execute(ex['inputs'],registry=registry)
        for signal in signals:
            a=torch.tensor(trace[signal.source].flat());b=torch.tensor(ex['targets'][signal.target].flat())
            worst=max(worst,float((a-b).abs().max()))
    return worst

def _discrete_report(model,program,examples,signals,result,decision,tolerance,mdl_weight,rank='order'):
    """A discrete backend's result in the shape `fit` returns, so callers do not branch."""
    # A failed discrete search has no error to report, and `None` rather than an
    # infinity keeps the report writable by `cli.write_json`, which forbids NaN
    # and infinities so an unrunnable number can never be recorded as a result.
    error=_exact_error(model.export(),examples,signals,model.registry) if result.solved else None
    return {'selection':decision.to_dict(),'training':[],'mdl_weight':mdl_weight,'rank':rank,
            'description_bits':float(model.description_cost().detach()),
            'pruned_description_bits':model.export().pruned().description_bits(model.registry) if result.solved else None,
            'freeze_events':[],'fully_frozen':bool(result.solved),
            'loss':0. if result.solved else None,'relaxed_loss':None,
            'exact_max_error':error,'exact_conformance':error is not None and error<=tolerance,
            'tolerance':tolerance,'discrete_result':result.to_dict()}

def fit(program,examples,signals,steps=300,lr=.05,freeze=True,registry=None,tolerance=.001,polish=200,mdl_weight=0.,
        mode='relax',validation=(),rollout_cost=0,select_options=None,ticks=1,settle_window=1,rank='order'):
    """`mode` picks the search backend; it defaults to the shipped one.

    `'relax'` is the gradient path this function has always run and is the
    default, so nothing shipped changes. `'auto'` measures the scaffold and the
    data and dispatches through `tcn.select.select_backend`, which may return
    `'enumerate'`, `'relax'` or `'hybrid'`. `'enumerate'` and `'hybrid'` force
    those backends. Whatever runs, the returned report has the same keys, plus a
    `selection` block recording what was decided and on what measurement.

    `validation` is used only by the discrete backends, and only because
    non-uniqueness is the norm rather than the exception: on a space with 2,464
    of 32,000 conforming programs, `enumerate_fit`'s returned program was
    measurably wrong on fresh episodes, and requiring exactness on a held-out
    split is what fixed it. Passing one costs a linear factor and buys a
    program that was checked off its training data.

    `rollout_cost` is how many environment steps one program evaluation
    consumes. It is zero for supervised examples already in hand, and it is the
    only property of the problem the selector cannot measure for itself.

    `mdl_weight` scales ARCHITECTURE section 8's `L_program_description`.

    It weights `SoftProgram.description_cost()`, the expected description length
    in bits of the pruned hardened program -- not `complexity()`, which is a
    softmax-weighted sum of `operator.cost` and therefore measures execution,
    where a module call is at exact parity with its inlined body. It defaults to
    zero so the shipped fixtures are unchanged; the term is bits against a probe
    loss, so a weight around 1e-5 is the scale at which it competes.

    `rank` is the same preference on the *discrete* path, where a gradient term
    has nothing to act on. `mdl_weight` only ever reached the relaxation loop;
    the discrete backends have carried `rank` in `order`/`description`/`cost`
    since `enumerate_fit` grew it, but `fit` did not pass it, so every discrete
    run took the first conforming program in enumeration order regardless of its
    size. It defaults to `'order'` so nothing shipped changes. `'description'`
    ranks by `Program.description_bits` of the pruned hardened program and
    `'cost'` by its `execution_cost`; both need the whole conforming set, so
    neither is compatible with a backend that stops at the first hit, and
    `mode='hybrid'` therefore rejects anything but `'order'`.

    A note the measurement forces: ranking can only choose *between* the
    conforming programs a scaffold admits. Where every program in the declared
    space has the same node count -- which is the case for every artifact this
    repository ships -- ranking is provably inert, and
    `research/program-length/RESULTS.md` carries the enumeration certificates
    that say so.
    """
    if not examples:raise ValueError('training examples required')
    if polish<0:raise ValueError('polish steps must be non-negative')
    if mdl_weight<0:raise ValueError('description weight must be non-negative')
    if rank not in {'order','description','cost'}:raise ValueError('unknown ranking '+str(rank))
    if mode not in {'relax','auto','enumerate','hybrid'}:raise ValueError('unknown search mode '+mode)
    if rank!='order' and mode=='hybrid':raise ValueError('ranking requires the full conforming set, which the hybrid backend does not build')
    program.validate_signals(signals)
    decision=None
    if mode!='relax':
        from .select import Decision,select_backend,hybrid_fit
        from .search import enumerate_fit
        opts=dict(select_options or {})
        decision=(select_backend(program,examples,signals,registry,tolerance,rollout_cost,**opts) if mode=='auto'
                  else Decision(mode,'backend forced by the caller',
                                space_size=math.prod(len(n.candidates) for n in program.nodes),
                                trainable_constants=tuple(program.trainable_constants)))
        if decision.mode in {'enumerate','hybrid'}:
            # Non-uniqueness is the norm, and a conforming program picked off the
            # training data alone was measured wrong on fresh episodes. Scoring
            # against training plus validation is the fix that was measured to work.
            scored=list(examples)+list(validation)
            constants=None
            if decision.mode=='hybrid':
                if rank!='order':raise ValueError('ranking requires the full conforming set, which the hybrid backend does not build')
                result,constants=hybrid_fit(program,scored,signals,registry,tolerance)
            else:
                # Route through the discrete backend rather than assuming the
                # feed-forward scorer. A program with `state` cannot be scored
                # feed-forward at all -- measured, `enumerate_fit` returns zero
                # conforming programs on a recurrent scaffold that
                # `enumerate_recurrent` certifies unique -- so dispatching on
                # `route` is a correctness matter, not a speed one.
                from .search import DiscreteProblem,route,solve
                problem=DiscreteProblem(program=program,examples=tuple(scored),signals=tuple(signals),
                                        ticks=ticks,settle_window=settle_window,tolerance=tolerance,registry=registry)
                result=(solve(problem,rank=rank) if route(problem) not in (None,'fit')
                        else enumerate_fit(program,scored,signals,registry,tolerance,rank=rank))
            if not result.solved and result.exhausted and mode=='auto':
                # An exhausted sweep that finds nothing is a completeness
                # certificate, not a budget failure: no program in the declared
                # space is exact on this data. That is worth reporting, and it is
                # also the cheapest possible way to be wrong -- the sweep cost is
                # the one the selector already measured -- so the run continues on
                # the relaxation, which can still return a best-effort program
                # under noisy or partially inconsistent supervision. `enumerate_fit`
                # has no minimum-Hamming objective, so this is the only route to
                # the partial-credit answer FINDINGS section 7 records.
                decision.notes=decision.notes+(
                    f'enumeration exhausted {result.space_size} programs and certified that none is '
                    f'exact within {tolerance}; falling through to the relaxation for a best effort',)
                decision.certified=False
                decision.mode='relax'
            else:
                model=SoftProgram(program,registry)
                if result.solved:
                    for name,index in result.selections.items():model.freeze(name,index)
                    if constants:
                        with torch.no_grad():
                            for k,v in constants.items():model.constants[k].copy_(v)
                    for p in model.constants.values():p.requires_grad_(False)
                return model,_discrete_report(model,program,examples,signals,result,decision,tolerance,mdl_weight,rank)
    model=SoftProgram(program,registry)
    if decision is not None and decision.scale_surrogates:model.scale_surrogates()
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    inputs={k:torch.stack([tensor(ex['inputs'][k]) for ex in examples]) for k,_ in program.inputs}
    targets={s.target:torch.stack([tensor(ex['targets'][s.target]) for ex in examples]) for s in signals}
    def loss_fn():
        _,_,trace=model(inputs,return_trace=True)
        return model.probe_loss(trace,targets,signals)
    history=[];torch.set_num_threads(1)
    for step in range(steps):
        optimizer.zero_grad();loss=loss_fn()+.001*(step/max(1,steps))*model.entropy()
        if mdl_weight:loss=loss+mdl_weight*model.description_cost()
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
    return model,{'selection':None if decision is None else decision.to_dict(),
                  'training':history,'mdl_weight':mdl_weight,'rank':rank,'description_bits':float(model.description_cost().detach()),
                  'pruned_description_bits':model.export().pruned().description_bits(model.registry),
                  'freeze_events':[asdict(x) for x in scheduler.events],'fully_frozen':len(model.frozen)==len(program.nodes) and all(not p.requires_grad for p in model.constants.values()),'loss':float(loss_fn().detach()),'relaxed_loss':float(loss_fn().detach()),'exact_max_error':error,'exact_conformance':error<=tolerance,'tolerance':tolerance}
