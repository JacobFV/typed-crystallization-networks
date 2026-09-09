"""Local computer kernel integrated with typed actions and deterministic replay state."""
import atexit
import hashlib
import json
import subprocess
from pathlib import Path
from tcn.generation import Generator,text_value,read_text,image_value,vector_value,Value,SCALAR,integer
from tcn.types import BOOL,product,setof
from generators.raster_text.generator import render_text
TEXT=text_value('',512).type;PATH=text_value('',128).type;KEY=integer(16,signed=False)
U8=integer(8,signed=False);U16=integer(16,signed=False);U32=integer(32,signed=False)
# --- panel interface (gated) -------------------------------------------------
# A narrow typed action argument, in the sense ARCHITECTURE section 1 already
# gives: a bounded integer. `tcn/policy.py:parameter_width` charges a numeric
# integer argument two policy parameters (mean, log sigma) and
# `numeric_bounds` reads the declared bit width, so `SLOT` is 4 values from 2
# parameters and `DIAL` is 16 values from 2 parameters. The shell interface's
# `write.text` is 1026 parameters scored by exact string equality; this is the
# same generator, the same kernel and the same reward mechanism with the action
# argument narrowed, which is the change `research/computer-capability/RESULTS.md`
# asked for. Neither type is `role="category"`: a category is sampled bit by bit
# and, being outside `Type.numeric`, admits no arithmetic, so a program could not
# compute the argument it wants to emit.
# `SLOT` wraps rather than erroring, so it is a 2-bit ring and `add(slot, 1)` is a
# total function on it: a program can advance the dial to the next slot without an
# out-of-range value, which is what lets a policy sweep the panel deterministically
# from the executed-action input `tcn/policy.py:action_inputs` already provides.
SLOT=integer(2,signed=False,overflow='wrap')
DIAL=integer(4,signed=False)
# The default shell menu, frozen as a literal. It used to be `tuple(action_schema)`,
# which would silently grow when the panel verbs were added to the schema; the
# recorded `available_actions` of every pre-existing configuration must not move.
SHELL_MENU=('wait','command','type','key','read','write')
PANEL_MENU=('wait','look','dial','commit')
PANEL_VERBS=frozenset(PANEL_MENU)-{'wait'}
PANEL_ROOT='/home/agent'
PANEL_OUT=PANEL_ROOT+'/out.txt'
PANEL_SLOTS=4
# Decoy keys, none of which begins with `t`.
PANEL_DECOY_KEYS=('note','memo','log','data','ref','aux')
# Every task key begins with `t` and no decoy key does, and the task keys have four
# different lengths, so the brand predicate is a single byte comparison while the
# digit's address is not a constant -- the same pair of perceptual problems the
# shell task posed, at a narrower action interface.
PANEL_TASK_KEYS=('tmp','task','tally','target')
PANEL_DIGITS=9                                 # task digit 0..8, so the answer 1..9 is in range
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

class Session:
    """One long-lived `engine/session.ts` process, addressed exactly like `bridge.ts`.

    Opt-in and off by default: `execute` uses it only when `configuration['session']`
    put one in `SESSIONS`. The transport is the only difference -- the same event
    log is applied to the same kernel in the same order under the same logical
    clock, and `research/credit-assignment/session_equivalence.py` checks the two
    payloads agree byte for byte. It exists because `bridge.ts` reboots the kernel
    for every transition, which costs ~1.9 s per acting step and puts a
    credit-assignment study out of reach.
    """
    def __init__(self):
        root=Path(__file__).parent/'engine'
        self.process=subprocess.Popen(['node','--import','tsx',str(root/'session.ts')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,cwd=root,bufsize=1)
        if json.loads(self.process.stdout.readline()).get('ready') is not True:raise RuntimeError('computer session did not start')
        atexit.register(self.close)
    def call(self,request):
        if self.process.poll() is not None:raise RuntimeError('computer session exited: '+(self.process.stderr.read() or '')[-2000:])
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('computer session closed: '+(self.process.stderr.read() or '')[-2000:])
        result=json.loads(line)
        if 'error' in result:raise RuntimeError('computer transition failed: '+str(result['error'])[-2000:])
        return result
    def close(self):
        if self.process.poll() is None:
            try:self.process.stdin.close()
            except Exception:pass
            try:self.process.wait(timeout=10)
            except Exception:self.process.kill()

SESSIONS={}

def session(name='default'):
    """The named live session, started on first use. Nothing calls this by default."""
    if name not in SESSIONS:SESSIONS[name]=Session()
    return SESSIONS[name]

def execute(seed,events,time,probe=None,transport=None):
    request={'seed':seed,'events':events,'time':time}
    if probe is not None:request['probe']=probe
    if transport is not None:return session(transport).call(request)
    root=Path(__file__).parent/'engine'
    cmd=['node','--import','tsx',str(root/'bridge.ts')]
    p=subprocess.run(cmd,input=json.dumps(request),text=True,capture_output=True,cwd=root,timeout=90)
    if p.returncode:raise RuntimeError('computer transition failed: '+p.stderr[-2000:])
    return json.loads(p.stdout)

class Implementation(Generator):
    action_schema={'wait':{},'command':{'text':TEXT},'type':{'text':TEXT},'key':{'code':KEY},'read':{'path':PATH},'write':{'path':PATH,'text':TEXT},
                   'look':{'slot':SLOT},'dial':{'value':DIAL},'commit':{}}
    def panel_setup(self,address,configuration):
        """The panel episode's hidden content, drawn on its own named random stream.

        Named streams are `Address.rng`'s reason for existing, so nothing here can
        move a draw any pre-existing configuration makes: `initialize` still takes
        the kernel seed from the default stream, in the same order, whether or not
        the panel is configured.

        Hand-initialisations, stated (AGENTS.md): every task key begins with `t`
        and no decoy key does, so `terminal[0] == 't'` is an exact predicate; the
        digit is always the last byte of a record whose length varies, so the
        address must be computed rather than constant; and the register starts at
        a value the agent did not choose, which is what gives a myopic policy
        something to be myopic about. Every one of the three is a property of the
        generated data, not of a model input, and none of them is the answer: the
        agent still has to find which slot holds the task record, read a digit out
        of raw bytes, and choose when to act.
        """
        rng=address.rng('panel')
        slots=int(configuration.get('panel_slots',PANEL_SLOTS))
        if not 2<=slots<=(1<<SLOT.bits):raise ValueError('panel_slots must fit the declared slot type')
        task=rng.randrange(slots);digit=rng.randrange(int(configuration.get('panel_digits',PANEL_DIGITS)))
        records=[]
        for i in range(slots):
            if i==task:records.append(f'{PANEL_TASK_KEYS[rng.randrange(len(PANEL_TASK_KEYS))]} = {digit}')
            else:records.append(f'{PANEL_DECOY_KEYS[rng.randrange(len(PANEL_DECOY_KEYS))]} = {rng.randrange(10)}')
        register=rng.randrange(1<<DIAL.bits) if configuration.get('panel_register') is None else int(configuration['panel_register'])
        return {'slots':slots,'task':task,'digit':digit,'records':records,'register':register,
                'paths':[f'{PANEL_ROOT}/slot{i}.txt' for i in range(slots)],'answer':digit+1}
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
        interface=configuration.get('interface','shell');panel=None
        if interface=='panel':
            panel=self.panel_setup(address,configuration)
            events=[{'kind':'write','path':p,'text':t,'time':0} for p,t in zip(panel['paths'],panel['records'])]
            spec={'root':PANEL_ROOT,'depth':0,'content_paths':list(panel['paths'])+[PANEL_OUT],'filesystem_capacity':0,'process_capacity':0}
        else:
            events=[{'kind':'write','path':'/home/agent/task.txt','text':configuration.get('document','count = 7'),'time':0}]
            spec=configuration.get('probe')
        request=self.probe_request(spec,configuration.get('objective',{}))
        transport=configuration.get('session')
        result=execute(seed,events,0,request,transport)
        state={'time':0.,'tick':0,'seed':seed,'events':events,'result':result,'buffer':'','interface':interface,'horizon':configuration.get('horizon',8),'objective':configuration.get('objective',{}),'screen_width':configuration.get('screen_width',128),'screen_height':configuration.get('screen_height',48)}
        if transport is not None:state['session']=transport
        if panel is not None:state['panel']=panel
        if spec:state['probe']=spec;state['probe_request']=request
        return state
    def advance(self,s,actions,dt,rng):
        changes=[];committed=False
        for a in actions:
            if s['interface']=='keyboard' and a.verb not in {'wait','type','key'}:raise ValueError('action unavailable at keyboard interface')
            if s['interface']!='panel' and a.verb in PANEL_VERBS:raise ValueError('action unavailable outside the panel interface')
            if s['interface']=='panel' and a.verb not in PANEL_MENU:raise ValueError('action unavailable at panel interface')
            if a.verb=='look':
                slot=int(a.arg('slot'))
                if 0<=slot<s['panel']['slots']:changes.append({'kind':'read','path':s['panel']['paths'][slot]})
            if a.verb=='dial':s['panel']['register']=int(a.arg('value'))
            if a.verb=='commit':
                committed=True;changes.append({'kind':'write','path':PANEL_OUT,'text':str(s['panel']['register'])})
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
        if changes:s['result']=execute(s['seed'],s['events'],s['time']+dt,request,s.get('session'))
        if s['interface']=='panel':
            # The reward is read back off the filesystem, not off the register: the
            # commit has to actually land in the file for it to count, exactly as the
            # shell task's exact-match reward is on file content. `commit` ends the
            # episode, so the reward arrives once, on the step the episode ends, and
            # every action that made it possible was taken strictly earlier.
            reward=0.
            if committed:
                row=self.probe_content(s,PANEL_OUT)
                reward=float(row is not None and row['content']==str(s['panel']['answer']))
            s['done']=committed or s['tick']+1>=s['horizon']
            return s,{'events':Value.of(integer(32,signed=False),len(changes))},{'goal':reward}
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
        if s['interface'] in {'shell','panel'}:observations['terminal']=text_value(str(text)[-1024:],4096)
        menu=('wait','type','key') if s['interface']=='keyboard' else PANEL_MENU if s['interface']=='panel' else SHELL_MENU
        snap=s['result']['snapshot']
        latents={'event_count':Value.of(integer(32,signed=False),len(s['events']))}
        probes={'state_counts':vector_value([len(snap['computers']),len(snap['packets']),len(s['result']['trajectory'])])}
        if s.get('probe'):
            extra_latents,extra_probes=self.state_view(s);latents.update(extra_latents);probes.update(extra_probes)
        if s.get('panel'):
            # Privileged panel state, probes and latents only. `answer` is the dense
            # target the staged arm supervises on; `showing_task` is the perceptual
            # predicate; `register` is state the agent's own `dial` changed and cannot
            # observe. `StepRecord.actor_view` reaches none of them.
            p=s['panel']
            latents['register']=Value.of(DIAL,p['register'])
            probes['answer']=Value.of(DIAL,p['answer']);probes['task_slot']=Value.of(SLOT,p['task'])
            probes['showing_task']=Value.of(BOOL,str(text)[:1]=='t')
        return observations,latents,probes,{'agent_0':menu}
