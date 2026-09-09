"""Does the tie-break matter? Every conforming policy program, scored held out.

FINDINGS section 14 records `enumerate_fit`'s pick being measurably wrong on fresh
episodes while the conforming set was large. The same question is asked here.
"""
import json,sys,time
sys.path.insert(0,'/home/brandonin/Documents/typed-crystallization-networks')
sys.path.insert(0,'/home/brandonin/Documents/typed-crystallization-networks/research/computer-capability')
import itertools
from tcn.operators import Registry
from tcn.search import evaluate,candidate_counts
import program as P, search_run as S, task as T
BASE='/home/brandonin/Documents/typed-crystallization-networks/research/computer-capability/out/'
train=S.policy_examples(T.load(BASE+'examples_train.json'))
test=S.policy_examples(T.load(BASE+'examples_test.json'))
r=Registry();prog=P.policy_program(r)
names=[n.name for n in prog.nodes];counts=candidate_counts(prog)
chosen=json.load(open(BASE+'search.json'))['policy']['enumeration']['selections']
by={n.name:n for n in prog.nodes}
rows=[];started=time.perf_counter()
for combination in itertools.product(*(range(c) for c in counts)):
    selections=dict(zip(names,combination))
    error=evaluate(prog,selections,train,P.policy_signals(),r,tolerance=.001)
    if error is None or error>.001: continue
    held=S.accuracy(prog,selections,test,P.policy_signals(),r)
    rows.append({'selections':selections,'test_accuracy':held,
                 'ppos':[by['ppos'].candidates[selections['ppos']].operator.name,
                         list(by['ppos'].candidates[selections['ppos']].sources)],
                 'brand_constant':by['brand'].candidates[selections['brand']].sources[1],
                 'is_the_returned_program':selections==chosen})
out={'conforming':len(rows),'generalizing':sum(r['test_accuracy']==1. for r in rows),
     'returned_program_test_accuracy':next(r['test_accuracy'] for r in rows if r['is_the_returned_program']),
     'test_accuracy_histogram':{str(round(v,3)):sum(1 for r in rows if round(r['test_accuracy'],3)==round(v,3))
                                for v in sorted({round(r['test_accuracy'],3) for r in rows})},
     'seconds':time.perf_counter()-started,'rows':rows}
open(BASE+'conforming_set.json','w').write(json.dumps(out,indent=2,sort_keys=True))
print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2,sort_keys=True))
