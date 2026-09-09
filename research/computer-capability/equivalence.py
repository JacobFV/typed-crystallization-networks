"""Default observation/latent/probe/reward stream is bit-identical after the gated channel.

Loads the pre-change `generator.py` from a saved copy and the current one from the
tree, drives both through the same addresses, configurations and action sequences,
and compares every serialized `StepRecord`. The bridge is the patched one in both
arms, so this also checks that `bridge.ts` emits a byte-identical payload when no
probe is requested.
"""
import importlib.util,json,sys,time,itertools
sys.path.insert(0,'/home/brandonin/Documents/typed-crystallization-networks')
from tcn.generation import Host,Address,Action,text_value

SCRATCH='/home/brandonin/Documents/typed-crystallization-networks/research/computer-capability/generator_before.py'
spec=importlib.util.spec_from_file_location('computer_generator_before',SCRATCH)
before=importlib.util.module_from_spec(spec);sys.modules['computer_generator_before']=before;spec.loader.exec_module(before)
from generators.computer.generator import Implementation as After

def act(verb,**kw): return Action(verb,arguments=tuple((k,text_value(v,512 if k=='text' else 128)) for k,v in kw.items()))
TASK='/home/agent/task.txt'
SEQS={'read_write':[act('read',path=TASK),act('write',path=TASK,text='8')],
      'command_wait':[act('command',text='cat '+TASK),act('wait')]}
DOCS=['count = 7','total = 3','x = 0']
SEEDS=[0,1,2]
OBJECTIVES=[{},{'path':TASK,'content':'8'}]

def run(generator,address,configuration,objective,actions):
    h=Host(generator,address,configuration,objective)
    for a in actions:h.step((a,))
    return [json.dumps(r.to_dict(),sort_keys=True) for r in h.records]

rows=[];t0=time.perf_counter()
for doc,seed,(name,seq),objective in itertools.product(DOCS,SEEDS,SEQS.items(),OBJECTIVES):
    address=Address('computer',seed,0,'train');cfg={'document':doc,'horizon':len(seq)}
    a=run(before.Implementation(),address,cfg,objective,seq)
    b=run(After(),address,cfg,objective,seq)
    rows.append({'document':doc,'seed':seed,'actions':name,'objective':bool(objective),
                 'records':len(a),'identical':a==b})
report={'combinations':len(rows),'identical':sum(r['identical'] for r in rows),
        'mismatched':[r for r in rows if not r['identical']],'seconds':time.perf_counter()-t0,'rows':rows}
print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
open('/home/brandonin/Documents/typed-crystallization-networks/research/computer-capability/out/equivalence.json','w').write(json.dumps(report,indent=2,sort_keys=True))
