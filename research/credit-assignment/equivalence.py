"""The default observation set and sampling stream of `generators/computer` are unchanged.

Loads the pre-change `generator.py` from `generator_before.py` (a copy taken before
this track edited the tree) and the current one, drives both through the same
addresses, configurations, action sequences and objectives, and compares every
serialized `StepRecord` field for field. Both arms run the *same* `bridge.ts`, so
this also checks that adding `session.ts` beside it changed nothing about the
default transport.

192 combinations by default, matching the precedent in
`research/computer-capability/equivalence.py`.
"""
import importlib.util,itertools,json,os,sys,time
ROOT='/home/brandonin/Documents/typed-crystallization-networks'
sys.path.insert(0,ROOT)
from tcn.generation import Host,Address,Action,text_value

spec=importlib.util.spec_from_file_location('computer_generator_before',ROOT+'/research/credit-assignment/generator_before.py')
before=importlib.util.module_from_spec(spec);sys.modules['computer_generator_before']=before;spec.loader.exec_module(before)
# `execute` resolves the engine from the module's own `__file__`; the copy lives
# elsewhere, so point it back at the one engine both arms must share.
before.__file__=ROOT+'/generators/computer/generator.py'
from generators.computer.generator import Implementation as After

def act(verb,**kw): return Action(verb,arguments=tuple((k,text_value(v,512 if k=='text' else 128)) for k,v in kw.items()))
TASK='/home/agent/task.txt'
SEQS={'read_write':[act('read',path=TASK),act('write',path=TASK,text='8')],
      'command_wait':[act('command',text='cat '+TASK),act('wait')],
      'write_command':[act('write',path=TASK,text='3'),act('command',text='ls /home/agent')]}
DOCS=['count = 7','total = 3','x = 0','accumulator = 9']
SEEDS=[0,1,2,3]
OBJECTIVES=[{},{'path':TASK,'content':'8'}]
SPLITS=['train','validation']
PROBES=[None,{'root':'/home/agent','depth':1,'content_paths':[TASK],'filesystem_capacity':8,'process_capacity':4}]

def run(generator,address,configuration,objective,actions):
    h=Host(generator,address,configuration,objective)
    for a in actions:h.step((a,))
    return [json.dumps(r.to_dict(),sort_keys=True) for r in h.records]

def main(limit=None,chunk=None,chunks=1):
    rows=[];t0=time.perf_counter()
    grid=list(itertools.product(DOCS,SEEDS,SEQS.items(),OBJECTIVES,SPLITS,PROBES))
    if limit:grid=grid[:limit]
    if chunk is not None:grid=grid[chunk::chunks]
    for doc,seed,(name,seq),objective,split,probe in grid:
        address=Address('computer',seed,0,split);cfg={'document':doc,'horizon':len(seq)}
        if probe:cfg['probe']=probe
        a=run(before.Implementation(),address,cfg,objective,seq)
        b=run(After(),address,cfg,objective,seq)
        rows.append({'document':doc,'seed':seed,'actions':name,'objective':bool(objective),'split':split,
                     'probe':bool(probe),'records':len(a),'identical':a==b})
        if a!=b:print('MISMATCH',rows[-1],flush=True)
    report={'combinations':len(rows),'identical':sum(r['identical'] for r in rows),
            'mismatched':[r for r in rows if not r['identical']],'seconds':time.perf_counter()-t0,'rows':rows}
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    name='equivalence.json' if chunk is None else f'equivalence_{chunk}.json'
    open(ROOT+'/research/credit-assignment/out/'+name,'w').write(json.dumps(report,indent=2,sort_keys=True))

def merge(chunks):
    rows=[]
    for i in range(chunks):
        rows+=json.load(open(f'{ROOT}/research/credit-assignment/out/equivalence_{i}.json'))['rows']
    report={'combinations':len(rows),'identical':sum(r['identical'] for r in rows),
            'mismatched':[r for r in rows if not r['identical']],'chunks':chunks,'rows':rows}
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    open(ROOT+'/research/credit-assignment/out/equivalence.json','w').write(json.dumps(report,indent=2))

if __name__=='__main__':
    if sys.argv[1:2]==['merge']:merge(int(sys.argv[2]))
    else:main(None,int(sys.argv[1]),int(sys.argv[2])) if len(sys.argv)>2 else main(int(sys.argv[1]) if len(sys.argv)>1 else None)
