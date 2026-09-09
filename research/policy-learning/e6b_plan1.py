"""Does enumerating action sequences buy anything over a one-step decision here?"""
import json, statistics, torch
from pl import Cfg, Counter, Runner, joint_program, freeze_choices, mpc_rollout
out=[]
for seed in range(4):
    c=Counter(); prog,reg=joint_program(policy_init='zero')
    r=Runner(prog,reg,Cfg(episodes=25,horizon=4,seed=seed,lr=.04,w_probe=1.,w_actor=0.,w_value=0.),c)
    r.train(log_every=25); freeze_choices(r); ex=r.model.export()
    for h in (4,8,16,32):
        for plan in (1,2,4):
            cc=Counter()
            ret=statistics.fmean(mpc_rollout(cc,ex,reg,0,10000+j,h,split='test',
                                             invert=bool(j%2),seed=0,plan_horizon=plan) for j in range(16))
            out.append({'seed':seed,'horizon':h,'plan_horizon':plan,'mean_return':ret,
                        'normalized':ret/h,'sequences_per_step':2**plan})
agg={}
for r in out: agg.setdefault((r['horizon'],r['plan_horizon']),[]).append(r['mean_return'])
for k,v in sorted(agg.items()):
    print('horizon %2d plan %2d  mean_return %.3f  normalized %.3f'%(k[0],k[1],statistics.fmean(v),statistics.fmean(v)/k[0]))
json.dump(out,open('research/policy-learning/out/e6b_plan1.json','w'),indent=1)
