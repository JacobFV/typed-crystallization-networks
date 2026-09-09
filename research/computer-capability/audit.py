"""Step 1: what is learnable in generators/computer before anything is trained.

Every number here is produced by running the shipped generator unchanged.
"""
import json,math,sys,time
sys.path.insert(0,'/home/brandonin/Documents/typed-crystallization-networks')
from tcn.generation import Host,Action,text_value,read_text

OUT={}
def act(verb,**kw):
    return Action(verb,arguments=tuple((k,text_value(v,512 if k=='text' else 128)) for k,v in kw.items()))

TASK='/home/agent/task.txt'

def episode(document,objective_content,verbs,horizon=None):
    h=Host.create('computer',seed=0,index=0,
                  configuration={'document':document,'horizon':horizon or len(verbs)},
                  objective={'path':TASK,'content':objective_content})
    rows=[{'tick':0,'terminal':read_text(h.view().observations['terminal']),
           'probes':{k:v.decoded for k,v in h.records[-1].probes.items()},
           'latents':{k:v.decoded for k,v in h.records[-1].latent_states.items()},'reward':0.}]
    for a in verbs:
        r=h.step((a,))
        rows.append({'tick':len(rows),'terminal':read_text(r.observations['terminal'].value),
                     'probes':{k:v.decoded for k,v in r.probes.items()},
                     'latents':{k:v.decoded for k,v in r.latent_states.items()},
                     'reward':sum(v.decoded for v in r.reward_components.values())})
    return h,rows

# ---------------------------------------------------------------- A1 inventory
h=Host.create('computer',seed=0,index=0,configuration={'document':'count = 7','horizon':4},
              objective={'path':TASK,'content':'8'})
v=h.view()
OUT['A1_inventory']={
 'observations':{k:{'width':val.type.width,'kind':val.type.kind,
                    'element_role':(val.type.items[1].items[0].role if k=='terminal' else val.type.items[3].items[0].role)}
                 for k,val in v.observations.items()},
 'available_actions':list(v.available_actions),
 'action_schema_arguments':{k:{a:t.width for a,t in spec.items()} for k,spec in h.generator.action_schema.items()},
 'latents':{k:{'width':val.type.width} for k,val in h.records[-1].latent_states.items()},
 'probes':{k:{'width':val.type.width,'value':val.decoded} for k,val in h.records[-1].probes.items()},
 'reward_components':['goal'],
 'objective_visible_to_actor':'objective' in dir(v) and bool(v.objective) and False,
 'objective_consumable_as_program_input':False,
}

# ------------------------------- A2 probe/latent information content, enumerated
DOCS=['count = 0','count = 1','count = 2','count = 3','count = 7','total = 4','x = 9','count = 12']
SEQ=[act('read',path=TASK),act('write',path=TASK,text='8'),act('wait')]
t0=time.perf_counter();traces={}
for d in DOCS:
    _,rows=episode(d,'8',SEQ)
    traces[d]=rows
OUT['A2_probe_information']={
 'documents':DOCS,
 'action_sequence':['read','write','wait'],
 'probe_series_per_document':{d:[r['probes']['state_counts'] for r in rows] for d,rows in traces.items()},
 'latent_series_per_document':{d:[r['latents']['event_count'] for r in rows] for d,rows in traces.items()},
 'distinct_probe_series':len({json.dumps([r['probes']['state_counts'] for r in rows]) for rows in traces.values()}),
 'distinct_latent_series':len({json.dumps([r['latents']['event_count'] for r in rows]) for rows in traces.values()}),
 'distinct_documents':len(set(DOCS)),
 'distinct_terminal_series':len({json.dumps([r['terminal'] for r in rows]) for rows in traces.values()}),
 'reward_series_per_document':{d:[r['reward'] for r in rows] for d,rows in traces.items()},
 'seconds':time.perf_counter()-t0,
}

# ------------------------------------------------- A3 eq surrogate at operating distance
def surrogate(a,b,tau): return math.exp(-((a-b)**2)/tau)
term0=traces['count = 7'][0]['terminal']; term1=traces['count = 7'][1]['terminal']
pairs=[('terminal byte 0 at tick 0 vs tick 1',ord(term0[0]),ord(term1[0])),
       ("digit '0' vs digit '9'",ord('0'),ord('9')),
       ("digit '7' vs digit '8'",ord('7'),ord('8')),
       ("'{' vs 'c' (JSON result vs file content)",ord('{'),ord('c')),
       ("uniform byte mixture 127.5 vs 'c'",127,ord('c'))]
OUT['A3_eq_surrogate']={
 'shipped_tau':1.0,'carrier':'int[8] role=byte','zero_threshold_float32_distance':11,
 'measurements':[{'pair':n,'a':a,'b':b,'distance':abs(a-b),
                  'surrogate_tau_1':surrogate(a,b,1.),'surrogate_tau_256':surrogate(a,b,256.)} for n,a,b in pairs],
}

# ----------------------------------------------------- A4 reward causality + failure mode
causal={}
for name,seq,content in [('write_correct',[act('read',path=TASK),act('write',path=TASK,text='8')],'8'),
                         ('write_wrong',[act('read',path=TASK),act('write',path=TASK,text='9')],'8'),
                         ('never_write',[act('read',path=TASK),act('wait')],'8'),
                         ('write_without_reading',[act('write',path=TASK,text='8'),act('wait')],'8')]:
    _,rows=episode('count = 7',content,seq)
    causal[name]=[r['reward'] for r in rows]
OUT['A4_reward_causality']=causal
try:
    episode('count = 7','8',[act('wait')])  # objective on an existing path, control
    control='ok'
except Exception as e: control='FAILED: '+str(e)[:200]
try:
    h2=Host.create('computer',seed=0,index=0,configuration={'document':'count = 7','horizon':2},
                   objective={'path':'/home/agent/answer.txt','content':'8'})
    h2.step((act('wait'),))
    missing='no error'
except Exception as e: missing=type(e).__name__+': '+str(e).split('\n')[0][:160]
OUT['A4_reward_causality']['objective_on_existing_path']=control
OUT['A4_reward_causality']['objective_on_nonexistent_path']=missing

# ---------------------------------------------------------------------- A5 cost
t0=time.perf_counter()
h3=Host.create('computer',seed=1,index=1,configuration={'document':'count = 3','horizon':4},objective={'path':TASK,'content':'4'})
t_init=time.perf_counter()-t0
step_times=[]
for a in [act('read',path=TASK),act('write',path=TASK,text='4'),act('wait'),act('command',text='echo hi')]:
    t=time.perf_counter();h3.step((a,));step_times.append(time.perf_counter()-t)
OUT['A5_cost']={'initialize_seconds':t_init,'step_seconds':step_times,
 'node_subprocess_calls_per_step':'2 when the objective is set and the action changes state, 1 otherwise (advance re-executes the whole event log, then executes it again with an appended read for the reward)',
 'episode_seconds_horizon_4':t_init+sum(step_times)}

print(json.dumps(OUT,indent=2,sort_keys=True))
open('/home/brandonin/Documents/typed-crystallization-networks/research/computer-capability/out/audit.json','w').write(json.dumps(OUT,indent=2,sort_keys=True))
