"""The discrete reference: what the reward-only search space actually contains."""
import json, itertools
def truth(k,a,b): return bool((k>>(2*int(a)+int(b)))&1)
# 32 contexts: 4 bits x goal. Only bits[0],bits[1],goal matter (fixed_inputs, depth 1).
ctx=[(b0,b1,g) for b0 in (0,1) for b1 in (0,1) for g in (0,1)]
def target(b0,b1,g): return (b0^b1)^g
reward_optimal=[];probe_optimal=[]
for k,j in itertools.product(range(16),repeat=2):
    zs=[int(truth(j,truth(k,b0,b1),g)) for b0,b1,g in ctx]
    ts=[target(*c) for c in ctx]
    if zs==ts: probe_optimal.append((k,j,'+'))
    if zs==ts or zs==[1-t for t in ts]: reward_optimal.append((k,j,'+' if zs==ts else '-'))
print('programs in the joint scaffold choice space:',16*16)
print('probe-optimal (z == target):',len(probe_optimal),probe_optimal)
print('reward-optimal up to readout sign:',len(reward_optimal),reward_optimal)
# how many (k,j) give a z that is *some* function of the context at all
const=sum(1 for k,j in itertools.product(range(16),repeat=2)
          if len({int(truth(j,truth(k,b0,b1),g)) for b0,b1,g in ctx})==1)
print('constant-z programs (readout can never beat chance):',const)
json.dump({'space':256,'probe_optimal':probe_optimal,'reward_optimal':reward_optimal,
           'constant_z':const},open('research/policy-learning/out/space.json','w'),indent=1)
