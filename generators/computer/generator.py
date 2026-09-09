"""Local computer kernel integrated with typed actions and deterministic replay state."""
import hashlib
import json
import subprocess
from pathlib import Path
from tcn.generation import Generator,text_value,read_text,image_value,vector_value,Value,SCALAR,integer
from tcn.types import BOOL,product,setof
from generators.raster_text.generator import render_text
TEXT=text_value('',512).type;PATH=text_value('',128).type;KEY=integer(16,signed=False)
U8=integer(8,signed=False);U16=integer(16,signed=False);U32=integer(32,signed=False)
# Capacity-declared relations over kernel state, in the shape ARCHITECTURE section 1
# already gives relations: sets of tuples. The leading index field is load-bearing
# exactly as in `generators/logic`'s `gates` channel -- a set is duplicate-free, and
# two directory entries can otherwise agree on every field.
FS_ENTRY=product(U32,U32,U32,U8,U32,U16)      # (index, parent, name hash, kind, size, mode)
PROC_ENTRY=product(U16,U16,U8,U32)            # (pid, ppid, state, executable hash)
KINDS={'file':0,'directory':1,'symlink':2}
STATES={'running':0,'sleeping':1,'stopped':2,'zombie':3}

def name_hash(text):
    """Stable 32-bit name identity. Computed here, not in the kernel, so the value
    does not depend on a JavaScript string hash and replays across interpreters."""
    return int.from_bytes(hashlib.sha256(text.encode('utf8')).digest()[:4],'little')

def execute(seed,events,time,probe=None):
    root=Path(__file__).parent/'engine'
    cmd=['node','--import','tsx',str(root/'bridge.ts')]
    request={'seed':seed,'events':events,'time':time}
    if probe is not None:request['probe']=probe
    p=subprocess.run(cmd,input=json.dumps(request),text=True,capture_output=True,cwd=root,timeout=90)
    if p.returncode:raise RuntimeError('computer transition failed: '+p.stderr[-2000:])
    return json.loads(p.stdout)

class Implementation(Generator):
    action_schema={'wait':{},'command':{'text':TEXT},'type':{'text':TEXT},'key':{'code':KEY},'read':{'path':PATH},'write':{'path':PATH,'text':TEXT}}
    def probe_request(self,spec,objective):
        """Bridge-side request for the gated privileged state view.

        `configuration['probe']` is absent by default, in which case this returns
        `None`, the bridge receives no `probe` key, and every observation, latent,
        probe, transition and reward is exactly what it was before this channel
        existed. The objective's own path is always read when the channel is on,
        because the reward is defined on it.
        """
        if not spec:return None
        paths=list(spec.get('content_paths',()))
        if objective.get('path') and objective['path'] not in paths:paths.append(objective['path'])
        return {'root':spec.get('root','/home/agent'),'depth':int(spec.get('depth',1)),'contents':paths}
    def initialize(self,address,configuration):
        seed=address.rng().randrange(2**31)
        events=[{'kind':'write','path':'/home/agent/task.txt','text':configuration.get('document','count = 7'),'time':0}]
        spec=configuration.get('probe')
        request=self.probe_request(spec,configuration.get('objective',{}))
        result=execute(seed,events,0,request)
        state={'time':0.,'tick':0,'seed':seed,'events':events,'result':result,'buffer':'','interface':configuration.get('interface','shell'),'horizon':configuration.get('horizon',8),'objective':configuration.get('objective',{}),'screen_width':configuration.get('screen_width',128),'screen_height':configuration.get('screen_height',48)}
        if spec:state['probe']=spec;state['probe_request']=request
        return state
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
        request=s.get('probe_request')
        for event in changes:event['time']=s['time']+dt;s['events'].append(event)
        if changes:s['result']=execute(s['seed'],s['events'],s['time']+dt,request)
        objective=s['objective'];reward=0.
        if objective.get('path'):
            row=self.probe_content(s,objective['path'])
            if row is None:
                # Default path, unchanged: append a read of the objective and run the
                # whole event log again. This raises if the objective names a file that
                # does not exist yet, so a create-a-file objective terminates the episode
                # before the agent can satisfy it.
                check=execute(s['seed'],s['events']+[{'kind':'read','path':objective['path'],'time':s['time']+dt}],s['time']+dt)
                reward=float(check['output'].get('content')==objective.get('content'))
            else:
                # Gated path: the objective's content already came back with the probe
                # view, from the same subprocess call. Same comparison on the same state,
                # one subprocess instead of two, and a missing file reads as unsatisfied
                # rather than raising.
                reward=float(row['content']==objective.get('content'))
        s['done']=s['tick']+1>=s['horizon'];return s,{'events':Value.of(integer(32,signed=False),len(changes))},{'goal':reward}
    def probe_content(self,s,path):
        if not s.get('probe'):return None
        return next((e for e in (s['result'].get('probe') or {}).get('contents',()) if e['path']==path),None)
    def state_view(self,s):
        """Typed privileged view of kernel state: filesystem and process relations,
        declared file contents, and objective satisfaction. Probes and latents only --
        `StepRecord.actor_view` cannot reach either, so the actor still sees exactly
        `terminal` and `pixels` and must compute everything else from them."""
        spec=s['probe'];raw=s['result'].get('probe') or {'files':[],'processes':[],'contents':[]}
        latents={'file_count':Value.of(U32,len(raw['files'])),'process_count':Value.of(U32,len(raw['processes']))}
        probes={}
        capacity=int(spec.get('filesystem_capacity',0))
        if capacity:
            # Deterministic truncation at the declared capacity, path-sorted, with the
            # true count in `file_count`: the type must not depend on episode content.
            kept=raw['files'][:capacity];order={e['path']:i for i,e in enumerate(kept)}
            rows=tuple((i,order.get(e['path'].rsplit('/',1)[0],i),name_hash(e['name']),KINDS.get(e['kind'],3),min(int(e['size']),2**32-1),int(e['mode'])&0xffff) for i,e in enumerate(kept))
            probes['filesystem']=Value.of(setof(FS_ENTRY,capacity),rows)
        capacity=int(spec.get('process_capacity',0))
        if capacity:
            kept=raw['processes'][:capacity]
            probes['processes']=Value.of(setof(PROC_ENTRY,capacity),tuple((int(p['pid'])&0xffff,int(p['ppid'])&0xffff,STATES.get(p['state'],7),name_hash(p['executable'])) for p in kept))
        capacity=int(spec.get('content_capacity',256))
        for i,entry in enumerate(raw['contents']):
            body=(entry['content'] or '').encode('utf8')[:capacity].decode('utf8','ignore')
            probes[f'content_{i}']=text_value(body,capacity);probes[f'content_{i}_present']=Value.of(BOOL,entry['content'] is not None)
        if s['objective'].get('path'):
            row=next((e for e in raw['contents'] if e['path']==s['objective']['path']),None)
            probes['goal_reached']=Value.of(BOOL,bool(row and row['content']==s['objective'].get('content')))
        return latents,probes
    def observe(self,s):
        output=s['result']['output'];text=output.get('stdout',output.get('content',json.dumps(output)))
        screen=render_text(str(text)[-512:]+'\n'+s['result']['prompt']+s['buffer'],s['screen_width'],s['screen_height'],11)
        observations={'pixels':image_value(screen)}
        if s['interface']=='shell':observations['terminal']=text_value(str(text)[-1024:],4096)
        menu=('wait','type','key') if s['interface']=='keyboard' else tuple(self.action_schema)
        snap=s['result']['snapshot']
        latents={'event_count':Value.of(integer(32,signed=False),len(s['events']))}
        probes={'state_counts':vector_value([len(snap['computers']),len(snap['packets']),len(s['result']['trajectory'])])}
        if s.get('probe'):
            extra_latents,extra_probes=self.state_view(s);latents.update(extra_latents);probes.update(extra_probes)
        return observations,latents,probes,{'agent_0':menu}
