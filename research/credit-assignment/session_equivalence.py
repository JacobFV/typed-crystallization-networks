"""`engine/session.ts` and `engine/bridge.ts` produce identical episodes.

The session transport is an optimisation and nothing else: it applies the same
event log to the same kernel in the same order under the same logical clock,
without rebooting between transitions. This drives the same addresses,
configurations and action sequences through both transports and compares every
serialized `StepRecord`, on the shell interface (where the default path is the
one every earlier track used) and on the panel interface.

Run: .venv/bin/python research/credit-assignment/session_equivalence.py [n]
"""
import itertools,json,sys,time
ROOT='/home/brandonin/Documents/typed-crystallization-networks'
sys.path.insert(0,ROOT)
from tcn.generation import Host,Action,text_value
from tcn.types import Value
from generators.computer.generator import SLOT,DIAL

def look(k):return Action('look',arguments=(('slot',Value.of(SLOT,k)),))
def dial(v):return Action('dial',arguments=(('value',Value.of(DIAL,v)),))
COMMIT=Action('commit',arguments=());WAIT=Action('wait',arguments=())
TASK='/home/agent/task.txt'
def act(verb,**kw):return Action(verb,arguments=tuple((k,text_value(v,512 if k=='text' else 128)) for k,v in kw.items()))

PANEL_SEQS={'find_then_commit':[look(0),look(1),look(2),dial(5),COMMIT],
            'commit_now':[COMMIT],
            'wait_look_dial':[WAIT,look(3),dial(0),dial(9),COMMIT],
            'never_commit':[look(1),WAIT,dial(2),WAIT,look(0),WAIT,WAIT,WAIT]}
SHELL_SEQS={'read_write':[act('read',path=TASK),act('write',path=TASK,text='8')],
            'command_wait':[act('command',text='cat '+TASK),act('wait')]}

def episode(configuration,objective,actions,index,split):
    h=Host.create('computer',seed=0,index=index,split=split,configuration=configuration,objective=objective)
    for a in actions:
        if h.records[-1].done:break
        h.step((a,))
    return [json.dumps(r.to_dict(),sort_keys=True) for r in h.records]

def main(limit=None):
    rows=[];t0=time.perf_counter()
    grid=[]
    for index,(name,seq),split in itertools.product(range(6),PANEL_SEQS.items(),('train','test')):
        grid.append(({'interface':'panel','horizon':8},{},seq,index,split,'panel/'+name))
    for index,(name,seq),doc in itertools.product(range(4),SHELL_SEQS.items(),('count = 7','x = 0')):
        grid.append(({'document':doc,'horizon':len(seq)},{'path':TASK,'content':'8'},seq,index,'train','shell/'+name))
    if limit:grid=grid[:limit]
    for configuration,objective,seq,index,split,name in grid:
        a=episode(configuration,objective,seq,index,split)
        b=episode(configuration|{'session':'equivalence'},objective,seq,index,split)
        rows.append({'case':name,'index':index,'split':split,'records':len(a),'identical':a==b})
        if a!=b:
            print('MISMATCH',rows[-1],flush=True)
            for x,y in zip(a,b):
                if x!=y:
                    dx,dy=json.loads(x),json.loads(y)
                    print('  keys differing:',[k for k in dx if json.dumps(dx[k],sort_keys=True)!=json.dumps(dy.get(k),sort_keys=True)])
                    break
    report={'combinations':len(rows),'identical':sum(r['identical'] for r in rows),
            'mismatched':[r for r in rows if not r['identical']],'seconds':time.perf_counter()-t0,'rows':rows}
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    open(ROOT+'/research/credit-assignment/out/session_equivalence.json','w').write(json.dumps(report,indent=2,sort_keys=True))

if __name__=='__main__':main(int(sys.argv[1]) if len(sys.argv)>1 else None)
