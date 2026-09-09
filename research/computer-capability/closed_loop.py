"""Closed loop: the exported exact program drives the live kernel, against baselines.

Nothing here is a proxy. Every number is the actual reward the shipped generator
returns for the action the frozen program chose, with the frozen exact `Agent` of
`tcn/agent.py`, on episodes whose file contents were never used to search the
program.
"""
import json,random,sys,time
sys.path.insert(0,'/home/brandonin/Documents/typed-crystallization-networks')
sys.path.insert(0,'/home/brandonin/Documents/typed-crystallization-networks/research/computer-capability')
from tcn.agent import Agent
from tcn.generation import Host,Action,text_value,read_text
from tcn.operators import Registry
from tcn.policy import ActionBinding
from tcn.runtime import save_program,load_program
from tcn.search import candidate_counts
from tcn.training import TrainConfig
import program as P
import task as T

BASE='/home/brandonin/Documents/typed-crystallization-networks/research/computer-capability/out/'
TASK_PATH=T.TASK_PATH

def templates(reader=None,write_path=TASK_PATH):
    return (Action('wait'),
            reader or Action('read',arguments=(('path',text_value(TASK_PATH,128)),)),
            Action('write',arguments=(('path',text_value(write_path,128)),('text',text_value('',512)))))

def configuration(reader=None,write_path=TASK_PATH,horizon=3):
    return TrainConfig(generator='computer',observations=('terminal',),targets=(),
                       action_templates=templates(reader,write_path),generator_config={},
                       horizon=horizon,dt=1.,
                       action_bindings=(ActionBinding(2,'text',P.TEXT,'text'),))

def episode_return(host):
    return sum(sum(v.decoded for v in r.reward_components.values()) for r in host.records)

def run_agent(program,registry,config,documents,objective_path=TASK_PATH,horizon=3,seed=0):
    rows=[]
    for i,(name,digit) in enumerate(documents):
        host=T.host(name,digit,horizon,objective_path=objective_path,seed=seed,index=1000+i,split='test')
        agent=Agent(program,registry,config,seed)
        agent.rollout(host,horizon,deterministic=True)
        rows.append({'document':T.document(name,digit),'target':T.target(digit),
                     'return':episode_return(host),
                     'actions':[a.verb for r in host.records for a in r.actions],
                     'written':[read_text(dict(a.arguments)['text']) for r in host.records for a in r.actions if a.verb=='write']})
    return {'episodes':len(rows),'mean_return':sum(r['return'] for r in rows)/len(rows),
            'max_return':horizon-1,'solved':sum(r['return']==horizon-1 for r in rows),'rows':rows}

def scripted(documents,choose,horizon=3,objective_path=TASK_PATH,label=''):
    """A baseline expressed as a fixed action rule over the same live episodes."""
    rows=[]
    for i,(name,digit) in enumerate(documents):
        host=T.host(name,digit,horizon,objective_path=objective_path,seed=0,index=1000+i,split='test')
        rng=random.Random(1000+i)
        for tick in range(horizon):
            if host.records[-1].done: break
            host.step((choose(tick,host,rng,objective_path),))
        rows.append({'document':T.document(name,digit),'return':episode_return(host)})
    return {'label':label,'episodes':len(rows),'mean_return':sum(r['return'] for r in rows)/len(rows),
            'max_return':horizon-1,'solved':sum(r['return']==horizon-1 for r in rows),'rows':rows}

def main():
    found=json.loads(open(BASE+'search.json').read())
    registry=Registry()
    transform=P.transform_program(registry)
    policy=P.policy_program(registry)
    tsel=found['transform']['enumeration']['selections']
    psel=found['policy']['enumeration']['selections']
    agent_program=P.agent_program(registry,transform,policy,tsel,psel)
    save_program(agent_program,BASE+'agent_program.json',registry)
    reloaded,reload_registry=load_program(BASE+'agent_program.json')
    report={'program':{'digest':agent_program.digest,'nodes':len(agent_program.nodes),
                       'pruned_nodes':len(agent_program.pruned().nodes),
                       'description_bits':agent_program.pruned().description_bits(registry),
                       'execution_cost':agent_program.pruned().execution_cost(registry),
                       'reload_digest_matches':reloaded.digest==agent_program.pruned().digest,
                       'transform_selection':tsel,'policy_selection':psel}}
    config=configuration()
    started=time.perf_counter()

    # ---------------------------------------------------------------- held-out documents
    report['A_unseen_documents']=run_agent(agent_program,registry,config,T.TEST_DOCUMENTS)
    report['A_training_documents']=run_agent(agent_program,registry,config,T.TRAIN_DOCUMENTS)

    # ------------------------------------------------------------------------- baselines
    def always_write(constant):
        def choose(tick,host,rng,path): return T.act('write',path=path,text=constant)
        return choose
    def read_then_constant(constant):
        def choose(tick,host,rng,path):
            return T.read_action() if tick==0 else T.act('write',path=path,text=constant)
        return choose
    def uniform(tick,host,rng,path):
        verb=rng.choice(['wait','read','write'])
        if verb=='wait': return T.wait_action()
        if verb=='read': return T.read_action()
        return T.act('write',path=path,text=str(rng.randrange(10)))
    report['B_baselines']=[
        scripted(T.TEST_DOCUMENTS,always_write('5'),label='always write "5"'),
        scripted(T.TEST_DOCUMENTS,read_then_constant('5'),label='read then write the modal digit "5"'),
        scripted(T.TEST_DOCUMENTS,uniform,label='uniform random verb, random digit'),
    ]
    # Ablation: the same scaffold at a random point of the searched space.
    rng=random.Random(7)
    names=[n.name for n in transform.nodes]
    random_t=dict(zip(names,[rng.randrange(c) for c in candidate_counts(transform)]))
    random_t.update({k:v for k,v in tsel.items() if k not in found['transform']['candidate_counts']})
    random_p=dict(psel)
    for k in found['policy']['candidate_counts']:
        random_p[k]=rng.randrange(len([n for n in policy.nodes if n.name==k][0].candidates))
    ablation=P.agent_program(registry,transform,policy,random_t,random_p)
    try:
        report['C_random_program_ablation']=run_agent(ablation,registry,config,T.TEST_DOCUMENTS)
    except Exception as error:
        report['C_random_program_ablation']={'failed':type(error).__name__+': '+str(error)[:200]}
    report['C_random_program_selection']={'transform':random_t,'policy':random_p}

    # ------------------------------------------------- unseen commands for the read step
    commands=['cat /home/agent/task.txt','head -n 1 /home/agent/task.txt',
              'tail -n 1 /home/agent/task.txt','grep = /home/agent/task.txt',
              'sed -n 1p /home/agent/task.txt','awk {print} /home/agent/task.txt',
              'wc -c /home/agent/task.txt']
    report['D_unseen_commands']=[]
    for command in commands:
        reader=Action('command',arguments=(('text',text_value(command,512)),))
        try:
            result=run_agent(agent_program,registry,configuration(reader),T.TEST_DOCUMENTS[:5])
        except Exception as error:
            result={'failed':type(error).__name__+': '+str(error)[:200]}
        report['D_unseen_commands'].append({'command':command}|result)

    # ---------------------------------------------------- unseen document formats
    formats=[('count=%d',None),('%d',None),('value: %d',None),('  total = %d',None),('answer -> %d',None)]
    report['E_unseen_formats']=[]
    for template,_ in formats:
        documents=[(template,d) for d in (1,4,6,8)]
        rows=[]
        for i,(fmt,digit) in enumerate(documents):
            text=fmt % digit
            host=Host.create('computer',seed=0,index=2000+i,split='test',
                             configuration={'document':text,'horizon':3},
                             objective={'path':TASK_PATH,'content':T.target(digit)})
            agent=Agent(agent_program,registry,config,0)
            try:
                agent.rollout(host,3,deterministic=True)
                rows.append({'document':text,'return':episode_return(host)})
            except Exception as error:
                rows.append({'document':text,'failed':type(error).__name__+': '+str(error)[:160]})
        good=[r for r in rows if 'return' in r]
        report['E_unseen_formats'].append({'format':template,'episodes':len(rows),
            'mean_return':(sum(r['return'] for r in good)/len(good)) if good else None,
            'failures':len(rows)-len(good),'rows':rows})

    # ------------------------------------------------------- unseen objective location
    report['F_unseen_objective_path']=[]
    for path in ['/home/agent/Desktop/notes.txt','/home/agent/task.txt']:
        try:
            result=run_agent(agent_program,registry,configuration(write_path=path),
                             T.TEST_DOCUMENTS[:4],objective_path=path)
        except Exception as error:
            result={'failed':type(error).__name__+': '+str(error)[:250]}
        report['F_unseen_objective_path'].append({'path':path}|result)

    report['seconds']=time.perf_counter()-started
    open(BASE+'closed_loop.json','w').write(json.dumps(report,indent=2,sort_keys=True,default=str))
    print(json.dumps({k:v for k,v in report.items() if not isinstance(v,list)},indent=2,sort_keys=True,default=str))

if __name__=='__main__': main()
