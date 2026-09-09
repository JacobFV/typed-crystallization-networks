"""The replay state is identical too, not only the record stream.

`equivalence.py` compares every serialized `StepRecord`. `Host.snapshot()['state']`
additionally carries the raw bridge payload in `state['result']`, so comparing it
checks that `bridge.ts` emits a byte-identical payload when no probe is requested --
the claim the gating is supposed to guarantee.
"""
import importlib.util,json,sys
sys.path.insert(0,'/home/brandonin/Documents/typed-crystallization-networks')
from tcn.generation import Host,Address,Action,text_value
BASE='/home/brandonin/Documents/typed-crystallization-networks/research/computer-capability/'
spec=importlib.util.spec_from_file_location('before',BASE+'generator_before.py')
before=importlib.util.module_from_spec(spec);spec.loader.exec_module(before)
from generators.computer.generator import Implementation as After
TASK='/home/agent/task.txt'
def act(v,**kw): return Action(v,arguments=tuple((k,text_value(x,512 if k=='text' else 128)) for k,x in kw.items()))
rows=[]
for objective in [{},{'path':TASK,'content':'8'}]:
    for document in ['count = 7','x = 0']:
        payloads=[]
        for generator in (before.Implementation(),After()):
            h=Host(generator,Address('computer',0,0,'train'),{'document':document,'horizon':2},objective)
            h.step((act('read',path=TASK),));h.step((act('write',path=TASK,text='8'),))
            snapshot=h.snapshot()
            payloads.append(json.dumps({k:snapshot[k] for k in ('state','inputs','records','configuration','objective')},sort_keys=True))
        rows.append({'document':document,'objective':bool(objective),
                     'state_inputs_and_records_identical':payloads[0]==payloads[1],
                     'bytes_compared':len(payloads[0])})
out={'combinations':len(rows),'identical':sum(r['state_inputs_and_records_identical'] for r in rows),'rows':rows}
open(BASE+'out/state_check.json','w').write(json.dumps(out,indent=2,sort_keys=True))
print(json.dumps(out,indent=2,sort_keys=True))
