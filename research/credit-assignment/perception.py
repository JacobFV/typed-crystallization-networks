"""Perception by enumeration against the probe channel, and the frozen model it yields.

Two small typed programs over `terminal`, searched exhaustively with
`tcn/search.py:enumerate_fit`, exactly as `research/computer-capability` searched
the shell task's address and shift. They are separate searches because they are
independent nodes; enumerating the joint space would be their product for no
extra information.

    transform :  terminal -> answer        (which byte, and what arithmetic)
    predicate :  terminal -> showing_task  (which byte, against which constant)

The examples come from recorded episodes driven through one fixed sweep, and the
targets come from `StepRecord.probes`, never from the generator's internal state.
Environment episodes are counted.

Run: .venv/bin/python research/credit-assignment/perception.py [episodes]
"""
import json,sys,time
ROOT='/home/brandonin/Documents/typed-crystallization-networks'
sys.path.insert(0,ROOT);sys.path.insert(0,ROOT+'/research/credit-assignment')
from tcn.types import BOOL,Value,integer,floating,product
from tcn.operators import Registry
from tcn.graph import Program,Node,Candidate,Signal,legal_candidates
from tcn.search import enumerate_fit,space_size
from panel import Counter,look,dial,COMMIT
from program import TERMINAL,LEN,DATA,BYTE,U8,F,_node,constant

SWEEP=[look(3),look(0),look(1),look(2),dial(0),COMMIT]

def transform_program():
    r=Registry()
    constants=[constant('one_len',LEN,1)]
    constants+=[constant(f'k{i}',LEN,v) for i,v in enumerate((2,3))]
    constants+=[constant(f'a{i}',LEN,v) for i,v in enumerate((0,4,7,8))]
    constants+=[constant(f'u{i}',F,float(v)) for i,v in enumerate((45.,46.,47.,48.))]
    ports={'length':LEN,'one_len':LEN}|{f'k{i}':LEN for i in range(2)}|{f'a{i}':LEN for i in range(4)}
    vports={'codef':F}|{f'u{i}':F for i in range(4)}
    nodes=[
      _node('length',LEN,[Candidate(r.resolve('project',(TERMINAL,),LEN,{'index':0}),('terminal',))],'perception',1),
      _node('data',DATA,[Candidate(r.resolve('project',(TERMINAL,),DATA,{'index':1}),('terminal',))],'perception',1),
      _node('pos',LEN,legal_candidates(r,('identity','sub'),ports,LEN,arities=(1,2)),'perception',2),
      _node('byte',BYTE,[Candidate(r.resolve('index',(DATA,LEN),BYTE),('data','pos'))],'perception',3),
      _node('unpacked',product(U8),[Candidate(r.resolve('unpack',(BYTE,),product(U8)),('byte',))],'perception',4),
      _node('code',U8,[Candidate(r.resolve('project',(product(U8),),U8,{'index':0}),('unpacked',))],'perception',5),
      _node('codef',F,[Candidate(r.resolve('decode',(U8,),F),('code',))],'perception',6),
      _node('answer',F,legal_candidates(r,('identity','sub'),vports,F,arities=(1,2)),'transform',7),
    ]
    return Program((('terminal',TERMINAL),),tuple(nodes),(('answer','answer'),),tuple(constants)).validate(r),r

def predicate_program():
    r=Registry()
    constants=[constant('zero_len',LEN,0)]
    constants+=[constant(f'a{i}',LEN,v) for i,v in enumerate((1,2,4,7))]
    constants+=[constant(f'b{i}',BYTE,ord(c)) for i,c in enumerate(('{','t','n','0',' ','='))]
    ports={'length':LEN,'zero_len':LEN}|{f'a{i}':LEN for i in range(4)}
    nodes=[
      _node('length',LEN,[Candidate(r.resolve('project',(TERMINAL,),LEN,{'index':0}),('terminal',))],'perception',1),
      _node('data',DATA,[Candidate(r.resolve('project',(TERMINAL,),DATA,{'index':1}),('terminal',))],'perception',1),
      _node('qpos',LEN,legal_candidates(r,('identity','sub'),ports,LEN,arities=(1,2)),'perception',2),
      _node('qbyte',BYTE,[Candidate(r.resolve('index',(DATA,LEN),BYTE),('data','qpos'))],'perception',3),
      _node('brand',BOOL,[Candidate(r.resolve('eq',(BYTE,BYTE)),('qbyte',f'b{i}')) for i in range(6)],'perception',4),
    ]
    return Program((('terminal',TERMINAL),),tuple(nodes),(('brand','brand'),),tuple(constants)).validate(r),r


def collect(counter,indices,split='train'):
    """Recorded episodes under one fixed sweep. Targets come only from `probes`."""
    transform_examples=[];predicate_examples=[]
    for index in indices:
        host=counter.create(index,split=split,horizon=len(SWEEP))
        records=[host.records[0]]
        for a in SWEEP:
            if host.records[-1].done:break
            records.append(counter.step(host,a))
        for record in records:
            terminal=record.observations['terminal'].value
            showing=bool(record.probes['showing_task'].decoded)
            predicate_examples.append({'inputs':{'terminal':terminal},
                                       'targets':{'reference_brand':Value.of(BOOL,showing)}})
            if showing:
                transform_examples.append({'inputs':{'terminal':terminal},
                                           'targets':{'reference_answer':Value.of(F,float(record.probes['answer'].decoded))}})
    return transform_examples,predicate_examples


def main(episodes=25):
    counter=Counter(session='perception');t0=time.perf_counter()
    train_t,train_p=collect(counter,range(episodes))
    held_t,held_p=collect(counter,range(20000,20000+16),split='test')
    tp,tr=transform_program();pp,pr=predicate_program()
    ts=(Signal('answer','reference_answer',('perception','transform'),F,'mse'),)
    ps=(Signal('brand','reference_brand',('perception',),BOOL,'mse'),)
    a=enumerate_fit(tp,train_t,ts,tr,tolerance=1e-6,rank='description')
    b=enumerate_fit(pp,train_p,ps,pr,tolerance=1e-6,rank='description')
    def named(program,result):
        if not result.selections:return None
        return {n.name:f'{n.candidates[result.selections[n.name]].operator.name}({",".join(n.candidates[result.selections[n.name]].sources)})'
                for n in program.nodes if len(n.candidates)>1}
    from tcn.search import evaluate as sweep_evaluate
    report={'episodes':episodes,'transform':a.to_dict()|{'space':space_size(tp),'program':named(tp,a),
              'held_out_max_error':sweep_evaluate(tp,a.selections,held_t,ts,tr) if a.selections else None,
              'train_examples':len(train_t)},
            'predicate':b.to_dict()|{'space':space_size(pp),'program':named(pp,b),
              'held_out_max_error':sweep_evaluate(pp,b.selections,held_p,ps,pr) if b.selections else None,
              'train_examples':len(train_p)},
            'environment_episodes':counter.episodes,'environment_steps':counter.steps,
            'seconds':time.perf_counter()-t0}
    print(json.dumps(report,indent=2,default=str))
    open(ROOT+'/research/credit-assignment/out/perception.json','w').write(json.dumps(report,indent=2,default=str))
    return report

if __name__=='__main__':main(int(sys.argv[1]) if len(sys.argv)>1 else 25)
