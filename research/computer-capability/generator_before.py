"""Verbatim copy of generators/computer/generator.py as it stood before the gated probe
channel, with only the engine path repointed at the real engine directory. Used by
equivalence.py to compare the default stream before and after."""
"""Local computer kernel integrated with typed actions and deterministic replay state."""
import json
import subprocess
from pathlib import Path
from tcn.generation import Generator,text_value,read_text,image_value,vector_value,Value,SCALAR,integer
from generators.raster_text.generator import render_text
TEXT=text_value('',512).type;PATH=text_value('',128).type;KEY=integer(16,signed=False)

def execute(seed,events,time):
    root=Path('/home/brandonin/Documents/typed-crystallization-networks/generators/computer/engine')
    cmd=['node','--import','tsx',str(root/'bridge.ts')]
    p=subprocess.run(cmd,input=json.dumps({'seed':seed,'events':events,'time':time}),text=True,capture_output=True,cwd=root,timeout=90)
    if p.returncode:raise RuntimeError('computer transition failed: '+p.stderr[-2000:])
    return json.loads(p.stdout)

class Implementation(Generator):
    action_schema={'wait':{},'command':{'text':TEXT},'type':{'text':TEXT},'key':{'code':KEY},'read':{'path':PATH},'write':{'path':PATH,'text':TEXT}}
    def initialize(self,address,configuration):
        seed=address.rng().randrange(2**31)
        events=[{'kind':'write','path':'/home/agent/task.txt','text':configuration.get('document','count = 7'),'time':0}]
        result=execute(seed,events,0)
        return {'time':0.,'tick':0,'seed':seed,'events':events,'result':result,'buffer':'','interface':configuration.get('interface','shell'),'horizon':configuration.get('horizon',8),'objective':configuration.get('objective',{}),'screen_width':configuration.get('screen_width',128),'screen_height':configuration.get('screen_height',48)}
    def advance(self,s,actions,dt,rng):
        changes=[]
        for a in actions:
            if s['interface']=='keyboard' and a.verb not in {'wait','type','key'}:raise ValueError('action unavailable at keyboard interface')
            if a.verb=='type':s['buffer']+=read_text(dict(a.arguments)['text'])
            if a.verb=='key':
                code=a.arg('code')
                if code==13:changes.append({'kind':'command','command':s['buffer']});s['buffer']=''
                elif code==8:s['buffer']=s['buffer'][:-1]
                else:raise ValueError('unsupported terminal key')
            if a.verb=='command':changes.append({'kind':'command','command':read_text(dict(a.arguments)['text'])})
            if a.verb=='write':changes.append({'kind':'write','path':read_text(dict(a.arguments)['path']),'text':read_text(dict(a.arguments)['text'])})
            if a.verb=='read':changes.append({'kind':'read','path':read_text(dict(a.arguments)['path'])})
        for event in changes:event['time']=s['time']+dt;s['events'].append(event)
        if changes:s['result']=execute(s['seed'],s['events'],s['time']+dt)
        objective=s['objective'];reward=0.
        if objective.get('path'):
            check=execute(s['seed'],s['events']+[{'kind':'read','path':objective['path'],'time':s['time']+dt}],s['time']+dt)
            reward=float(check['output'].get('content')==objective.get('content'))
        s['done']=s['tick']+1>=s['horizon'];return s,{'events':Value.of(integer(32,signed=False),len(changes))},{'goal':reward}
    def observe(self,s):
        output=s['result']['output'];text=output.get('stdout',output.get('content',json.dumps(output)))
        screen=render_text(str(text)[-512:]+'\n'+s['result']['prompt']+s['buffer'],s['screen_width'],s['screen_height'],11)
        observations={'pixels':image_value(screen)}
        if s['interface']=='shell':observations['terminal']=text_value(str(text)[-1024:],4096)
        menu=('wait','type','key') if s['interface']=='keyboard' else tuple(self.action_schema)
        snap=s['result']['snapshot']
        return observations,{'event_count':Value.of(integer(32,signed=False),len(s['events']))},{'state_counts':vector_value([len(snap['computers']),len(snap['packets']),len(s['result']['trajectory'])])},{'agent_0':menu}
