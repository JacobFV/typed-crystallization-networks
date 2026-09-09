"""One learning arm, one seed. Writes `out/arm_<name>_s<seed>.json`.

Arms
    reward          reward only, gamma = 0.95, neutral start, perception declared
    myopic          the same, gamma = 0 -- the learner that cannot assign credit
    reward_g50      the same at gamma = 0.5, straddling the exact 0.394 threshold
    staged          probe-supervise the searched perception, crystallize, then reward
    reward_percept  reward only with the perception choices live (gradient boundary)
    reward_sweep_given  reward only, with the sweeping sub-action supplied
    flat            reward only with a state-independent policy (structure ablation)
    probe_only      supervision only, no actor term -- the F-init control

Run: .venv/bin/python research/credit-assignment/arms.py <arm> <seed> [episodes]
"""
import json,sys,time
ROOT='/home/brandonin/Documents/typed-crystallization-networks'
sys.path.insert(0,ROOT);sys.path.insert(0,ROOT+'/research/credit-assignment')
import torch
from panel import Counter
from run import Runner,Cfg,freeze_perception

CHECKPOINTS=(50,100,200,400,800,1200)

ARMS={
 'reward':        dict(discount=.95),
 'myopic':        dict(discount=0.),
 'reward_g50':    dict(discount=.5),
 'reward_percept':dict(discount=.95,perception=True),
 'flat':          dict(discount=.95,policy_state=False),
 'probe_only':    dict(discount=.95,perception=True,w_actor=0.,w_value=0.,w_entropy=0.,w_probe=1.),
 'staged':        dict(discount=.95,perception=True),
 'staged_enum':   dict(discount=.95,perception=True),
 'dial_explores': dict(discount=.95,dial_explores=True),
 # The sub-action supplied rather than searched: `next_slot` has one candidate, the
 # sweep. This is the ceiling the reward-only arm is measured against -- it answers
 # "given the sub-action, does reward alone finish the task?" and it is the
 # non-modular twin of section 8's macro library.
 'reward_sweep_given': dict(discount=.95,slot_pool=False),
}

def run(arm,seed,episodes):
    counter=Counter(session=f'{arm}{seed}');t0=time.perf_counter()
    cfg=Cfg(episodes=episodes,seed=seed,**ARMS[arm])
    if arm=='staged_enum':
        # Stage 1 is enumeration against the probes, not gradient descent: the
        # address choice on this observation gets `grad is None` (FINDINGS section
        # 23), so a gradient supervision stage cannot learn it and staging it that
        # way would measure the gradient boundary again rather than the handoff.
        import perception as P
        from tcn.graph import Signal
        from tcn.search import enumerate_fit
        from tcn.types import BOOL
        from program import F
        train_t,train_p=P.collect(counter,range(25))
        tp,tr=P.transform_program();pp,pr=P.predicate_program()
        a=enumerate_fit(tp,train_t,(Signal('shift','reference_answer',('perception','transform'),F,'mse'),),tr,tolerance=1e-6,rank='description')
        b=enumerate_fit(pp,train_p,(Signal('brand','reference_brand',('perception',),BOOL,'mse'),),pr,tolerance=1e-6,rank='description')
        chosen={}
        for program,result in ((tp,a),(pp,b)):
            for node in program.nodes:
                if len(node.candidates)>1:
                    c=node.candidates[result.selections[node.name]]
                    chosen[node.name]=f'{c.operator.name}({",".join(c.sources)})'
        runner=Runner(cfg,counter)
        pinned={}
        for node,logits in zip(runner.program.nodes,runner.model.choices):
            if node.name in chosen and len(node.candidates)>1:
                names=[f'{c.operator.name}({",".join(c.sources)})' for c in node.candidates]
                if chosen[node.name] in names:
                    with torch.no_grad():logits[names.index(chosen[node.name])]=20.
                    runner.model.freeze(node.name);pinned[node.name]=chosen[node.name]
        if set(pinned)!=set(chosen):
            raise RuntimeError(f'enumerated selection did not install cleanly: {sorted(set(chosen)-set(pinned))}')
        runner.params=[p for p in runner.model.choices if p.requires_grad]+list(runner.model.constants.parameters())
        for p in runner.model.parameters():p.requires_grad_(False)
        for p in runner.params:p.requires_grad_(True)
        runner.optimizer=torch.optim.Adam(runner.params,lr=cfg.lr)
        runner.cfg.index_offset=0
        curve=runner.train(log_every=50,evaluate_at=set(c for c in CHECKPOINTS if c<=episodes))
        report={'stage1_episodes':counter.episodes,'enumerated':pinned,
                'transform_conforming':a.conforming,'predicate_conforming':b.conforming}
    elif arm=='staged':
        # stage 1: probe supervision only, no reward term at all
        stage1=Cfg(episodes=100,seed=seed,perception=True,w_actor=0.,w_value=0.,w_entropy=0.,w_probe=1.)
        runner=Runner(stage1,counter);runner.train(log_every=50)
        frozen=freeze_perception(runner)
        runner.cfg=cfg;runner.cfg.episodes=episodes;runner.cfg.index_offset=100
        curve=runner.train(log_every=50,evaluate_at=set(CHECKPOINTS))
        report={'stage1_episodes':100,'frozen':frozen}
    else:
        runner=Runner(cfg,counter)
        curve=runner.train(log_every=50,evaluate_at=set(c for c in CHECKPOINTS if c<=episodes))
        report={}
    final=runner.evaluate(64)
    final_stochastic=runner.evaluate(64,stochastic=True)
    report|={'arm':arm,'seed':seed,'episodes':episodes,'curve':curve,'final':final,
             'final_stochastic':final_stochastic,
             'program':runner.report(),'history':runner.history,
             'environment_episodes':counter.episodes,'environment_steps':counter.steps,
             'seconds':time.perf_counter()-t0}
    path=f'{ROOT}/research/credit-assignment/out/arm_{arm}_s{seed}.json'
    open(path,'w').write(json.dumps(report,indent=2,default=str))
    print(json.dumps({k:report[k] for k in ('arm','seed','final','final_stochastic','environment_episodes','seconds')},default=str))
    return report

if __name__=='__main__':
    arm=sys.argv[1];seed=int(sys.argv[2]);episodes=int(sys.argv[3]) if len(sys.argv)>3 else 800
    run(arm,seed,episodes)
