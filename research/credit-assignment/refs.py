"""Every baseline, on the real generator, measured not asserted.

Constant policies (`always_wait`, `look_only`, `commit_now`), uniform random, the
neutral typed sampler `tcn/policy.py` actually starts from, the myopic reference
(`commit_now` is exactly the gamma = 0 optimum), a non-myopic but uninformed
reference (`dial_then_commit`), and the exact oracle. Then the same oracle/myopic
pair swept over the horizon, against `analysis.py`'s exact values.

Run: .venv/bin/python research/credit-assignment/refs.py [episodes]
"""
import json,statistics,sys,time
ROOT='/home/brandonin/Documents/typed-crystallization-networks'
sys.path.insert(0,ROOT);sys.path.insert(0,ROOT+'/research/credit-assignment')
import panel
from panel import Counter,measure,POLICIES

def main(episodes=64):
    counter=Counter(session='refs');t0=time.perf_counter()
    indices=range(10000,10000+episodes)
    rows=[measure(counter,name,indices,split='test') for name in POLICIES]
    sweep=[]
    for horizon in (1,2,3,4,6,8):
        small=range(10000,10000+min(episodes,48))
        sweep.append({'horizon':horizon,
                      'oracle':measure(counter,'oracle',small,split='test',horizon=horizon)['mean'],
                      'commit_now':measure(counter,'commit_now',small,split='test',horizon=horizon)['mean'],
                      'uniform_random':measure(counter,'uniform_random',small,split='test',horizon=horizon)['mean'],
                      'neutral_typed':measure(counter,'neutral_typed',small,split='test',horizon=horizon)['mean'],
                      'episodes_each':len(small)})
    report={'horizon':panel.HORIZON,'episodes':episodes,'policies':rows,'horizon_sweep':sweep,
            'environment_episodes':counter.episodes,'environment_steps':counter.steps,
            'seconds':time.perf_counter()-t0}
    print(json.dumps(report,indent=2))
    open(ROOT+'/research/credit-assignment/out/refs.json','w').write(json.dumps(report,indent=2))

if __name__=='__main__':main(int(sys.argv[1]) if len(sys.argv)>1 else 64)
