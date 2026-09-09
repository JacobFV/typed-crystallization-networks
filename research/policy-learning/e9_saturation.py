"""Track 2 measured policy logits reaching +/-2.5 within ~30 episodes. Timescale here."""
import json, statistics, torch
from pl import Cfg, Counter, Runner, joint_program
rows=[]
for name,init,probe in (('reward_only','zero',0.),('shipped_probe','oracle',1.)):
    for seed in range(8):
        c=Counter(); prog,reg=joint_program(policy_init=init)
        cfg=Cfg(episodes=1,horizon=4,seed=seed,lr=.04,w_probe=probe,w_actor=1.,w_value=.5,
                w_entropy=.01,choice_bias12=(init=='oracle'))
        r=Runner(prog,reg,cfg,c)
        trace=[]
        for ep in range(0,300):
            r.cfg.episodes=1; r.cfg.index_offset=ep; r.train(log_every=0)
            with torch.no_grad():
                rw,_,_=r.rollout(900000+ep)
                trace.append(float(rw[0]['logits'].abs().max()))
        rows.append({'arm':name,'seed':seed,'trace':trace})
agg={}
for x in rows: agg.setdefault(x['arm'],[]).append(x['trace'])
out={}
for arm,ts in agg.items():
    out[arm]={str(e):round(statistics.fmean(t[e] for t in ts),3) for e in (0,10,20,30,50,100,200,299)}
    firsts=[next((i for i,v in enumerate(t) if v>=2.5), None) for t in ts]
    out[arm]['episodes_to_|logit|>=2.5']=[f if f is not None else '>300' for f in firsts]
print(json.dumps(out,indent=1))
json.dump({'summary':out,'raw':rows},open('research/policy-learning/out/e9_saturation.json','w'))
