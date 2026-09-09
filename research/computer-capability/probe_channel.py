"""The gated probe channel, and what it carries that `state_counts` does not."""
import json,sys,time
sys.path.insert(0,'/home/brandonin/Documents/typed-crystallization-networks')
from tcn.generation import Host,Action,text_value,read_text

TASK='/home/agent/task.txt'
PROBE={'root':'/home/agent','depth':1,'filesystem_capacity':24,'process_capacity':8,
       'content_capacity':64,'content_paths':[TASK]}
def act(verb,**kw): return Action(verb,arguments=tuple((k,text_value(v,512 if k=='text' else 128)) for k,v in kw.items()))

def episode(document,target,actions,probe=PROBE):
    cfg={'document':document,'horizon':len(actions)}
    if probe:cfg['probe']=probe
    h=Host.create('computer',seed=0,index=0,configuration=cfg,objective={'path':TASK,'content':target})
    rows=[h.records[-1]]
    for a in actions:rows.append(h.step((a,)))
    return h,rows

def summarize(r):
    out={}
    for k,v in r.probes.items():
        if k.startswith('content_') and not k.endswith('_present'):out[k]=read_text(v)
        elif v.type.kind=='set':out[k]=len(v.decoded)
        else:out[k]=v.decoded
    return out|{'latents':{k:v.decoded for k,v in r.latent_states.items()},
                'reward':sum(v.decoded for v in r.reward_components.values())}

t0=time.perf_counter()
report={}
h,rows=episode('count = 7','8',[act('read',path=TASK),act('write',path=TASK,text='8'),act('command',text='mkdir /home/agent/work')])
report['trace']=[summarize(r) for r in rows]
report['probe_types']={k:{'kind':v.type.kind,'width':v.type.width,'capacity':v.type.capacity} for k,v in rows[0].probes.items()}
report['actor_view_keys']=sorted(h.view().observations)
report['privileged_keys_absent_from_actor_view']=sorted(set(rows[0].probes)|set(rows[0].latent_states))
report['leak_check']=[k for k in report['privileged_keys_absent_from_actor_view'] if k in h.view().observations]

# information content, the same comparison the audit ran on `state_counts`
docs=['count = 0','count = 3','count = 7','total = 4','x = 9']
series={}
for d in docs:
    _,rs=episode(d,'8',[act('read',path=TASK)])
    series[d]={'state_counts':[r.probes['state_counts'].decoded for r in rs],
               'content_0':[read_text(r.probes['content_0']) for r in rs],
               'filesystem_size':[len(r.probes['filesystem'].decoded) for r in rs]}
report['information_content']={
 'documents':docs,
 'distinct_state_counts_series':len({json.dumps(v['state_counts']) for v in series.values()}),
 'distinct_content_0_series':len({json.dumps(v['content_0']) for v in series.values()}),
 'series':series}
report['seconds']=time.perf_counter()-t0
print(json.dumps(report,indent=2,sort_keys=True))
open('/home/brandonin/Documents/typed-crystallization-networks/research/computer-capability/out/probe_channel.json','w').write(json.dumps(report,indent=2,sort_keys=True))
