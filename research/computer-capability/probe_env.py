"""Measure what the computer generator actually exposes, and what it costs."""
import json,sys,time
sys.path.insert(0,'/home/brandonin/Documents/typed-crystallization-networks')
from tcn.generation import Host,Action,text_value,read_text

t0=time.perf_counter()
h=Host.create('computer',seed=0,index=0,configuration={'document':'count = 7','horizon':6},objective={'path':'/home/agent/task.txt','content':'8'})
t_init=time.perf_counter()-t0
v=h.view()
print('init seconds',round(t_init,3))
print('observation keys',sorted(v.observations))
for k,val in v.observations.items(): print(' ',k,'width',val.type.width)
print('available actions',v.available_actions)
print('terminal',repr(read_text(v.observations['terminal'])[:200]))
print('probes',{k:val.decoded for k,val in h.records[-1].probes.items()})

def act(verb,**kw):
    return Action(verb,arguments=tuple((k,text_value(v,512 if k=='text' else 128)) for k,v in kw.items()))

steps=[('command',{'text':'cat /home/agent/task.txt'}),
       ('read',{'path':'/home/agent/task.txt'}),
       ('write',{'path':'/home/agent/task.txt','text':'8'}),
       ('wait',{}),
       ('command',{'text':'echo hello'}),
       ('command',{'text':'ls /home/agent'})]
for verb,kw in steps:
    t0=time.perf_counter()
    r=h.step((act(verb,**kw),))
    dt=time.perf_counter()-t0
    txt=read_text(r.observations['terminal'].value) if 'terminal' in r.observations else ''
    print(f'--- {verb} {kw} sec={dt:.2f} reward={ {k:x.decoded for k,x in r.reward_components.items()} } probes={ {k:x.decoded for k,x in r.probes.items()} } transition={ {k:x.decoded for k,x in r.transition.items()} }')
    print('    terminal:',repr(txt[:200]))
