"""Random points of the same searched space, driven through the same live episodes.

An agent whose program raises mid-episode scores the return it had accumulated, which
is zero: a crash is a failure to act, not a missing measurement.
"""
import json,random,sys,time
sys.path.insert(0,'/home/brandonin/Documents/typed-crystallization-networks')
sys.path.insert(0,'/home/brandonin/Documents/typed-crystallization-networks/research/computer-capability')
from tcn.agent import Agent
from tcn.operators import Registry
from tcn.search import candidate_counts
import program as P, task as T, closed_loop as C
BASE='/home/brandonin/Documents/typed-crystallization-networks/research/computer-capability/out/'

registry=Registry()
transform=P.transform_program(registry);policy=P.policy_program(registry)
found=json.load(open(BASE+'search.json'))
searched={'transform':found['transform']['enumeration']['selections'],
          'policy':found['policy']['enumeration']['selections']}
free_t=set(found['transform']['candidate_counts']);free_p=set(found['policy']['candidate_counts'])
config=C.configuration()
rows=[];started=time.perf_counter()
rng=random.Random(11)
for arm in range(6):
    tsel=dict(searched['transform']);psel=dict(searched['policy'])
    for n in transform.nodes:
        if n.name in free_t: tsel[n.name]=rng.randrange(len(n.candidates))
    for n in policy.nodes:
        if n.name in free_p: psel[n.name]=rng.randrange(len(n.candidates))
    program=P.agent_program(registry,transform,policy,tsel,psel)
    returns=[];failures=0
    for i,(name,digit) in enumerate(T.TEST_DOCUMENTS[:5]):
        host=T.host(name,digit,3,seed=0,index=3000+i,split='test')
        agent=Agent(program,registry,config,0)
        try: agent.rollout(host,3,deterministic=True)
        except Exception: failures+=1
        returns.append(C.episode_return(host))
    rows.append({'arm':arm,'transform':{k:tsel[k] for k in free_t},'policy':{k:psel[k] for k in free_p},
                 'episodes':len(returns),'mean_return':sum(returns)/len(returns),
                 'crashed_episodes':failures})
out={'arms':len(rows),'episodes_per_arm':5,'max_return':2,
     'mean_return_over_all_arms':sum(r['mean_return'] for r in rows)/len(rows),
     'arms_detail':rows,'seconds':time.perf_counter()-started}
open(BASE+'ablation.json','w').write(json.dumps(out,indent=2,sort_keys=True))
print(json.dumps(out,indent=2,sort_keys=True))
