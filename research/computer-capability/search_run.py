"""Synthesis of the two searched regions, with the discrete reference and baselines.

Both arms search the identical candidate space: `tcn.search.enumerate_fit` walks it
exhaustively and certifies uniqueness where it exhausts; `tcn.synthesis.fit` relaxes
it. Constant and random baselines sit beside every number, and every space size is
enumerated rather than asserted.
"""
import json,random,sys,time
sys.path.insert(0,'/home/brandonin/Documents/typed-crystallization-networks')
sys.path.insert(0,'/home/brandonin/Documents/typed-crystallization-networks/research/computer-capability')
import torch
from tcn.operators import Registry
from tcn.types import Value
from tcn.search import enumerate_fit,evaluate,space_size,candidate_counts
from tcn.synthesis import fit
from tcn.learning import SoftProgram,tensor
import program as P
import task as T

BASE='/home/brandonin/Documents/typed-crystallization-networks/research/computer-capability/out/'

def transform_examples(payload):
    rows=[]
    for episode in payload['episodes']:
        for row in episode['rows']:
            if row['reference_byte'] is None: continue
            rows.append({'inputs':{'terminal':Value.from_dict(row['terminal'])},
                         'targets':{'reference_byte':Value.of(P.U8,row['reference_byte'])},
                         'label':episode['document']})
    return rows

def policy_examples(payload):
    rows=[]
    for episode in payload['episodes']:
        for row in episode['rows']:
            rows.append({'inputs':{'terminal':Value.from_dict(row['terminal']),'action':T.one_hot(row['previous'])},
                         'targets':{'reference_action':T.one_hot(row['reference'])},
                         'label':f"{episode['document']}@{row['tick']}"})
    return rows

def accuracy(program,selections,examples,signals,registry):
    hits=0
    for ex in examples:
        error=evaluate(program,selections,[ex],signals,registry,tolerance=.001)
        hits+= error is not None and error<=.001
    return hits/len(examples)

def random_baseline(program,examples,signals,registry,draws=400,seed=0):
    rng=random.Random(seed);counts=candidate_counts(program);names=[n.name for n in program.nodes]
    hits=0
    for _ in range(draws):
        selections=dict(zip(names,[rng.randrange(c) for c in counts]))
        error=evaluate(program,selections,examples,signals,registry,tolerance=.001)
        if error is not None and error<=.001: hits+=1
    return {'draws':draws,'conforming':hits,'rate':hits/draws}

def gradient_report(program,examples,signals,registry,watch=()):
    model=SoftProgram(program,registry)
    inputs={k:torch.stack([tensor(ex['inputs'][k]) for ex in examples]) for k,_ in program.inputs}
    targets={s.target:torch.stack([tensor(ex['targets'][s.target]) for ex in examples]) for s in signals}
    _,_,trace=model(inputs,return_trace=True)
    loss=model.probe_loss(trace,targets,signals)
    loss.backward()
    out={'loss':float(loss.detach()),
         'choice_gradient_max_abs':{n.name:(None if p.grad is None else float(p.grad.abs().max()))
                                    for n,p in zip(program.nodes,model.choices) if len(n.candidates)>1}}
    out['relaxed_node_values']={k:[round(float(x),6) for x in trace[k].flatten()[:6].tolist()] for k in watch if k in trace}
    return out

def run(program,train,test,signals,name,steps=300,max_programs=1<<20):
    registry=Registry()
    report={'name':name,'space_size':space_size(program),
            'candidate_counts':{n.name:len(n.candidates) for n in program.nodes if len(n.candidates)>1},
            'train_examples':len(train),'test_examples':len(test)}
    started=time.perf_counter()
    found=enumerate_fit(program,train,signals,registry,tolerance=.001,rank='description',max_programs=max_programs)
    report['enumeration']=found.to_dict()
    report['enumeration']['solution_density']=found.conforming/found.space_size if found.exhausted else None
    if found.solved:
        report['enumeration']['train_accuracy']=accuracy(program,found.selections,train,signals,registry)
        report['enumeration']['test_accuracy']=accuracy(program,found.selections,test,signals,registry)
        report['enumeration']['test_max_error']=evaluate(program,found.selections,test,signals,registry)
        report['enumeration']['selected_candidates']={
            n.name:{'operator':n.candidates[found.selections[n.name]].operator.name,
                    'sources':list(n.candidates[found.selections[n.name]].sources)}
            for n in program.nodes if len(n.candidates)>1}
    report['enumeration_seconds']=time.perf_counter()-started
    report['random_baseline']=random_baseline(program,train,signals,registry)
    started=time.perf_counter()
    report['initial_gradients']=gradient_report(program,train,signals,Registry(),
                                                watch=('brand','byte','pbyte','value','shift'))
    try:
        model,gradient=fit(program,train,signals,steps=steps,registry=Registry(),tolerance=.001,polish=0)
        selections=model.selections()
        report['gradient']={k:v for k,v in gradient.items() if k in
                            {'relaxed_loss','exact_max_error','exact_conformance','fully_frozen','tolerance'}}
        report['gradient']['selections']=selections
        report['gradient']['train_accuracy']=accuracy(program,selections,train,signals,Registry())
        report['gradient']['test_accuracy']=accuracy(program,selections,test,signals,Registry())
        report['gradient']['agrees_with_enumeration']=(selections==found.selections) if found.solved else None
    except Exception as error:
        report['gradient']={'failed':type(error).__name__+': '+str(error)[:300]}
    report['gradient_seconds']=time.perf_counter()-started
    return report

if __name__=='__main__':
    train=T.load(BASE+'examples_train.json');test=T.load(BASE+'examples_test.json')
    registry=Registry()
    out={}
    tp=P.transform_program(registry,encode=False)
    tr,te=transform_examples(train),transform_examples(test)
    # Constant baseline: the best single byte written regardless of the observation.
    targets=[ex['targets']['reference_byte'].decoded for ex in tr]
    tt=[ex['targets']['reference_byte'].decoded for ex in te]
    out['transform_constant_baseline']={
        'best_constant_train_accuracy':max(targets.count(v) for v in set(targets))/len(targets),
        'best_constant_test_accuracy':max(tt.count(v) for v in set(tt))/len(tt),
        'distinct_train_targets':len(set(targets)),'distinct_test_targets':len(set(tt))}
    out['transform']=run(tp,tr,te,P.signals(),'transform')

    pp=P.policy_program(registry)
    pr,pe=policy_examples(train),policy_examples(test)
    labels=[ex['targets']['reference_action'].decoded.index(1.) for ex in pr]
    lt=[ex['targets']['reference_action'].decoded.index(1.) for ex in pe]
    out['policy_constant_baseline']={
        'best_constant_train_accuracy':max(labels.count(v) for v in set(labels))/len(labels),
        'best_constant_test_accuracy':max(lt.count(v) for v in set(lt))/len(lt),
        'train_label_counts':{str(v):labels.count(v) for v in sorted(set(labels))}}
    out['policy']=run(pp,pr,pe,P.policy_signals(),'policy')

    open(BASE+'search.json','w').write(json.dumps(out,indent=2,sort_keys=True,default=str))
    print(json.dumps(out,indent=2,sort_keys=True,default=str))
