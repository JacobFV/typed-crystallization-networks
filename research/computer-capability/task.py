"""The task, its episodes, and the supervised examples extracted from them.

TASK. `/home/agent/task.txt` holds one line, `<name> = <n>` with `n` a single
decimal digit. The objective is satisfied when that file's whole content is the
successor digit, `str(n+1)`. The reward is the shipped one: the kernel reads the
file and compares its bytes.

WHY IT IS CAUSAL RATHER THAN A LOOKUP. At tick 0 the terminal shows the JSON
result of the setup write -- `{"path": "/home/agent/task.txt", "bytes": 9}` --
which carries the file's *length* and not its content. The digit is only visible
after the agent has itself issued a read, so an agent that writes at tick 0
cannot know what to write. `n` varies per episode, so no constant text is right
more than once in nine; and `name` varies in length, so no constant byte address
finds the digit. Reward responds to the agent's own write and to nothing else:
measured in `out/audit.json` as write-correct 1.0, write-wrong 0.0, never-write
0.0.
"""
import json,sys,time
sys.path.insert(0,'/home/brandonin/Documents/typed-crystallization-networks')
from tcn.generation import Host,Action,text_value,read_text
from tcn.types import Value,floating,product

TASK_PATH='/home/agent/task.txt'
F=floating()
ACTION=product(F,F,F)
# Template order is load-bearing: index 0 is `wait`, which is also the value of
# `previous` before the first action, so "no action yet" and "just waited" are the
# same observation and the policy never sees an uninitialised port.
TEMPLATES=('wait','read','write')

# Training names and digits. Names differ in length so that a constant byte address
# cannot locate the digit; digits differ so that no constant text can be written.
TRAIN_DOCUMENTS=[('n',3),('ab',5),('cnt',0),('total',6),('counter',2)]
# Held out: names never seen, digits never seen, and both lengths inside and
# outside the trained range.
TEST_DOCUMENTS=[('k',7),('xy',1),('idx',8),('amount',4),('accumulator',0),
                ('q',5),('zz',2),('sum',6),('length',3),('registers',7)]

def document(name,digit): return f'{name} = {digit}'
def target(digit): return str(digit+1)

def act(verb,**kw):
    return Action(verb,arguments=tuple((k,text_value(v,512 if k=='text' else 128)) for k,v in kw.items()))

def read_action(): return act('read',path=TASK_PATH)
def write_action(text): return act('write',path=TASK_PATH,text=text)
def wait_action(): return act('wait')

def host(name,digit,horizon=3,probe=None,objective_path=TASK_PATH,seed=0,index=0,split='train'):
    configuration={'document':document(name,digit),'horizon':horizon}
    if probe: configuration['probe']=probe
    return Host.create('computer',seed=seed,index=index,split=split,configuration=configuration,
                       objective={'path':objective_path,'content':target(digit)})

def scripted_episode(name,digit,reader=None,horizon=3):
    """The reference trajectory: read, write the successor, wait."""
    h=host(name,digit,horizon)
    actions=[reader or read_action(),write_action(target(digit)),wait_action()][:horizon]
    for a in actions: h.step((a,))
    return h,actions

def one_hot(i): return Value.of(ACTION,tuple(1. if j==i else 0. for j in range(3)))

def examples_from(h,actions):
    """One example per decision point: the observation and the previous action index
    the actor would have had, with the reference action and the reference byte."""
    rows=[]
    previous=0
    for tick,action in enumerate(actions):
        record=h.records[tick]
        terminal=record.observations['terminal'].value
        text=read_text(terminal)
        verb=action.verb
        rows.append({'tick':tick,'terminal':terminal.to_dict(),'previous':previous,
                     'terminal_text':text,'reference':TEMPLATES.index(verb),
                     'reference_byte':(ord(read_text(dict(action.arguments)['text'])) if verb=='write' else None)})
        previous=TEMPLATES.index(verb)
    return rows

def collect(documents,path,reader=None,horizon=3):
    started=time.perf_counter();out=[]
    for name,digit in documents:
        h,actions=scripted_episode(name,digit,reader,horizon)
        rows=examples_from(h,actions)
        out.append({'name':name,'digit':digit,'document':document(name,digit),'target':target(digit),
                    'return':sum(sum(v.decoded for v in r.reward_components.values()) for r in h.records),
                    'rows':rows})
    payload={'documents':[list(d) for d in documents],'episodes':out,'seconds':time.perf_counter()-started}
    open(path,'w').write(json.dumps(payload))
    return payload

def load(path): return json.loads(open(path).read())

if __name__=='__main__':
    base='/home/brandonin/Documents/typed-crystallization-networks/research/computer-capability/out/'
    train=collect(TRAIN_DOCUMENTS,base+'examples_train.json')
    test=collect(TEST_DOCUMENTS,base+'examples_test.json')
    print(json.dumps({'train_episodes':len(train['episodes']),'train_seconds':train['seconds'],
                      'train_returns':[e['return'] for e in train['episodes']],
                      'test_episodes':len(test['episodes']),'test_seconds':test['seconds'],
                      'test_returns':[e['return'] for e in test['episodes']],
                      'sample_terminals':[r['terminal_text'] for r in train['episodes'][0]['rows']]},indent=2))
