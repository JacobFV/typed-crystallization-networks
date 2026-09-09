"""Check the local harness against the shipped fixture before trusting it."""
import json, time, torch
from pl import Cfg, Counter, Runner, joint_program, reference_returns

t0=time.time()
c=Counter()
prog,reg=joint_program(policy_init='oracle',value_head='constant')
cfg=Cfg(episodes=160,horizon=4,seed=0,lr=.04,w_pred=0.,w_probe=1.,w_actor=1.,
        w_value=.5,w_entropy=.01,choice_bias12=True)
r=Runner(prog,reg,cfg,c)
r.train(log_every=32)
print('history tail:',json.dumps(r.history[-3:],indent=1))
print('eval return (64 held-out):',r.evaluate(64))
print('choices:',json.dumps(r.choice_report(),indent=1))
print('env episodes consumed by training+eval:',c.episodes,'steps:',c.steps)
c2=Counter()
print('references:',reference_returns(c2,range(10000,10064)))
print('reference env episodes:',c2.episodes)
print('wall',round(time.time()-t0,1),'s')
