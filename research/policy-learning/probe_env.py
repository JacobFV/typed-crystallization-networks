"""Establish the task's exact structure before measuring anything on it."""
import json
from tcn.generation import Host,Action
from tcn.types import BOOL,Value

SET={'depth':1,'table':6,'fixed_inputs':True}
def ep(i,invert,horizon=4,split='train',seed=0):
    h=Host.create('logic',seed=seed,index=i,split=split,configuration=SET|{'horizon':horizon},objective={'invert':invert})
    return h

# 1. context and reward structure
rows=[]
for i in range(8):
    inv=bool(i%2)
    h=ep(i,inv)
    v=h.view()
    bits=v.observations['bits'].decoded; goal=v.observations['goal'].decoded
    tgt=h.records[0].probes['target'].decoded; gate=h.records[0].probes['gate'].decoded
    rs=[]
    for t in range(4):
        rec=h.step((Action('answer',arguments=(('value',Value.of(BOOL,True)),)),),1.)
        rs.append(sum(x.decoded for x in rec.reward_components.values()))
    rows.append({'i':i,'bits':bits,'goal':goal,'target':tgt,'gate':gate,'parity':bits[0]^bits[1],'rewards_always_true':rs,'available':v.available_actions})
print(json.dumps(rows,indent=1))

# 2. does the observation change within an episode?
h=ep(0,False)
o0=h.view().observations['bits'].decoded
h.step((Action('answer',arguments=(('value',Value.of(BOOL,True)),)),),1.)
print('bits constant within episode:',o0==h.view().observations['bits'].decoded)

# 3. distinct contexts over 256 train episodes
seen={}
for i in range(256):
    inv=bool(i%2); h=ep(i,inv); v=h.view()
    key=(v.observations['bits'].decoded,v.observations['goal'].decoded)
    seen[key]=seen.get(key,0)+1
print('distinct (bits,goal) contexts in 256 train episodes:',len(seen))
print(sorted(seen.items())[:20])
