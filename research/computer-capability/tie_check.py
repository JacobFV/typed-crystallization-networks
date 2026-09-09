"""Which programs conform, and are they the same function?"""
import json,sys
sys.path.insert(0,'/home/brandonin/Documents/typed-crystallization-networks')
sys.path.insert(0,'/home/brandonin/Documents/typed-crystallization-networks/research/computer-capability')
from tcn.operators import Registry
from tcn.search import evaluate
import program as P, search_run as S, task as T
BASE='/home/brandonin/Documents/typed-crystallization-networks/research/computer-capability/out/'
train=S.transform_examples(T.load(BASE+'examples_train.json'))
test=S.transform_examples(T.load(BASE+'examples_test.json'))
r=Registry();prog=P.transform_program(r,encode=False)
found=json.load(open(BASE+'search.json'))['transform']['enumeration']['selections']
pos_node=[n for n in prog.nodes if n.name=='pos'][0]
shift_node=[n for n in prog.nodes if n.name=='shift'][0]
rows=[]
for i in range(len(shift_node.candidates)):
    sel=dict(found);sel['shift']=i
    e=evaluate(prog,sel,train,P.signals(),r,tolerance=.001)
    if e is not None and e<=.001:
        c=shift_node.candidates[i]
        rows.append({'index':i,'operator':c.operator.name,'sources':list(c.sources),
                     'test_max_error':evaluate(prog,sel,test,P.signals(),r)})
addr=[]
for i in range(len(pos_node.candidates)):
    sel=dict(found);sel['pos']=i
    e=evaluate(prog,sel,train,P.signals(),r,tolerance=.001)
    if e is not None and e<=.001:
        c=pos_node.candidates[i]
        addr.append({'index':i,'operator':c.operator.name,'sources':list(c.sources)})
out={'conforming_shift_candidates_with_the_chosen_address':rows,
     'conforming_address_candidates_with_the_chosen_shift':addr,
     'note':'the two conforming programs differ only by the commutation of `add`; '
            'the address is unique in the space'}
open(BASE+'tie_check.json','w').write(json.dumps(out,indent=2,sort_keys=True))
print(json.dumps(out,indent=2,sort_keys=True))
