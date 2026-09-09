"""One learning arm, one seed. Writes `out/arm_<name>_s<seed>.json`.

Arms
    reward          reward only, gamma = 0.95, neutral start, perception declared
    myopic          the same, gamma = 0 -- the learner that cannot assign credit
    reward_g50      the same at gamma = 0.5, straddling the exact 0.394 threshold
    staged          probe-supervise the searched perception, crystallize, then reward
    reward_percept  reward only with the perception choices live (gradient boundary)
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
}

def run(arm,seed,episodes):
    counter=Counter(session=f'{arm}{seed}');t0=time.perf_counter()
    cfg=Cfg(episodes=episodes,seed=seed,**ARMS[arm])
    if arm=='staged':
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
    report|={'arm':arm,'seed':seed,'episodes':episodes,'curve':curve,'final':final,
             'program':runner.report(),'history':runner.history,
             'environment_episodes':counter.episodes,'environment_steps':counter.steps,
             'seconds':time.perf_counter()-t0}
    path=f'{ROOT}/research/credit-assignment/out/arm_{arm}_s{seed}.json'
    open(path,'w').write(json.dumps(report,indent=2,default=str))
    print(json.dumps({k:report[k] for k in ('arm','seed','final','environment_episodes','seconds')},default=str))
    return report

if __name__=='__main__':
    arm=sys.argv[1];seed=int(sys.argv[2]);episodes=int(sys.argv[3]) if len(sys.argv)>3 else 800
    run(arm,seed,episodes)
