"""The panel task: action library, reference policies, and an exact abstract model.

Nothing here is a model input. The reference policies below read privileged state
(`Host.state['panel']`) because a reference is allowed to; the typed programs in
`program.py` see `terminal` and the executed-action ports and nothing else.

The panel configuration of `generators/computer` is:

    configuration = {'interface': 'panel', 'horizon': H, 'session': <name or absent>}

    observations       pixels, terminal            (unchanged in kind from `shell`)
    latents            event_count, register
    probes             state_counts, content_i, answer, task_slot, showing_task
    actions            wait | look(slot: int[2] wrap) | dial(value: int[4]) | commit
    reward             one component `goal`, 1.0 on the commit step iff
                       /home/agent/out.txt holds str(task digit + 1)
    termination        commit, or the horizon

`look` reads slot k's record into the terminal; `dial` sets a register the agent
cannot observe; `commit` writes the register to the objective file and ends the
episode. Exactly one slot holds `t<key> = <digit>`; the others hold `<key> = <digit>`
with a key that does not begin with `t`.
"""
from __future__ import annotations
import math,random,statistics
import sys
ROOT='/home/brandonin/Documents/typed-crystallization-networks'
if ROOT not in sys.path:sys.path.insert(0,ROOT)
from tcn.generation import Host,Action
from tcn.types import Value
from tcn.policy import ActionBinding
from generators.computer.generator import SLOT,DIAL,PANEL_SLOTS,PANEL_DIGITS

HORIZON=6
CONFIG={'interface':'panel','horizon':HORIZON}
WAIT=Action('wait',arguments=())
COMMIT=Action('commit',arguments=())
def look(k):return Action('look',arguments=(('slot',Value.of(SLOT,int(k))),))
def dial(v):return Action('dial',arguments=(('value',Value.of(DIAL,int(v))),))
# Template order is the policy's categorical order and is fixed everywhere.
TEMPLATES=(WAIT,look(0),dial(0),COMMIT)
BINDINGS=(ActionBinding(1,'slot',SLOT,'slot_params'),ActionBinding(2,'value',DIAL,'dial_params'))
VERBS=('wait','look','dial','commit')


class Counter:
    """Environment episodes and environment steps actually consumed."""
    def __init__(self,session=None):self.episodes=0;self.steps=0;self.session=session
    def create(self,index,seed=0,split='train',horizon=HORIZON,**extra):
        self.episodes+=1
        cfg=dict(CONFIG)|{'horizon':horizon}|extra
        if self.session is not None:cfg['session']=self.session
        return Host.create('computer',seed=seed,index=index,split=split,configuration=cfg)
    def step(self,host,action,dt=1.):
        self.steps+=1
        return host.step((action,),dt)


# ---------------------------------------------------------------------------
# neutral typed sampling: what a freshly initialised typed policy actually emits
# ---------------------------------------------------------------------------

def neutral_distribution(t,samples=None):
    """Exact P(value) for `sample_typed(t, [0., 0.])`, the zero-parameter policy.

    `tcn/policy.py` draws u ~ N(mean, exp(logstd)) and returns
    round(lo + (tanh(u)+1)/2*(hi-lo)) clamped to [lo, hi]. At mean = logstd = 0
    that is a fixed distribution over the integers in range, computed here in
    closed form from the normal CDF rather than sampled.
    """
    lo,hi=0.,float((1<<t.bits)-1)
    phi=lambda z:.5*(1+math.erf(z/math.sqrt(2)))
    out={}
    for k in range(int(lo),int(hi)+1):
        edges=[]
        for x in (k-.5,k+.5):
            z=2*(x-lo)/(hi-lo)-1
            if z<=-1:edges.append(-math.inf)
            elif z>=1:edges.append(math.inf)
            else:edges.append(math.atanh(z))
        out[k]=phi(edges[1])-phi(edges[0])
    return out

SLOT_NEUTRAL=neutral_distribution(SLOT)
DIAL_NEUTRAL=neutral_distribution(DIAL)


# ---------------------------------------------------------------------------
# reference policies
# ---------------------------------------------------------------------------

def policy_commit_now(host,t,rng,memory):
    """The myopic reference: the only action with nonzero immediate expected reward."""
    return COMMIT,memory

def policy_wait(host,t,rng,memory):return WAIT,memory
def policy_look_only(host,t,rng,memory):return look(t%PANEL_SLOTS),memory

def policy_uniform(host,t,rng,memory):
    verb=rng.randrange(4)
    if verb==0:return WAIT,memory
    if verb==1:return look(rng.randrange(1<<SLOT.bits)),memory
    if verb==2:return dial(rng.randrange(1<<DIAL.bits)),memory
    return COMMIT,memory

def _draw(distribution,rng):
    u=rng.random();acc=0.
    for k,p in distribution.items():
        acc+=p
        if u<acc:return k
    return max(distribution)

def policy_neutral_typed(host,t,rng,memory):
    """Uniform over templates, typed arguments from the zero-parameter sampler."""
    verb=rng.randrange(4)
    if verb==0:return WAIT,memory
    if verb==1:return look(_draw(SLOT_NEUTRAL,rng)),memory
    if verb==2:return dial(_draw(DIAL_NEUTRAL,rng)),memory
    return COMMIT,memory

def policy_dial_then_commit(host,t,rng,memory):
    """Non-myopic but uninformed: guess the answer, then commit. 1/9 by construction."""
    if memory is None:return dial(rng.randrange(1,PANEL_DIGITS+1)),'dialled'
    return COMMIT,memory

def policy_oracle(host,t,rng,memory):
    """Privileged reference. Sweeps the panel from slot 3, then dials and commits."""
    p=host.state['panel']
    order=[(3+i)%PANEL_SLOTS for i in range(PANEL_SLOTS)]
    stage,i=memory if isinstance(memory,tuple) else ('look',0)
    if stage=='look':
        if order[i]==p['task']:return look(order[i]),('dial',i)
        return look(order[i]),('look',i+1)
    if stage=='dial':return dial(p['answer']),('commit',i)
    return COMMIT,('done',i)

def policy_oracle_myopic_first(host,t,rng,memory):
    """The oracle's plan with the myopic first move spliced in, for the gap table."""
    if t==0:return COMMIT,memory
    return policy_oracle(host,t,rng,memory)

POLICIES={'commit_now':policy_commit_now,'always_wait':policy_wait,'look_only':policy_look_only,
          'uniform_random':policy_uniform,'neutral_typed':policy_neutral_typed,
          'dial_then_commit':policy_dial_then_commit,'oracle':policy_oracle}


def rollout(counter,policy,index,seed=0,split='train',horizon=HORIZON,rng=None):
    rng=rng or random.Random(index)
    host=counter.create(index,seed=seed,split=split,horizon=horizon)
    total=0.;memory=None;trace=[]
    for t in range(horizon):
        if host.records[-1].done:break
        action,memory=policy(host,t,rng,memory)
        record=counter.step(host,action)
        reward=sum(v.decoded for v in record.reward_components.values())
        total+=reward;trace.append((action.verb,reward))
        if record.done:break
    return total,trace,host


def measure(counter,name,indices,seed=0,split='test',horizon=HORIZON,repeats=1):
    policy=POLICIES[name];totals=[]
    for r in range(repeats):
        for i in indices:
            total,_,_=rollout(counter,policy,i,seed=seed,split=split,horizon=horizon,
                              rng=random.Random(1000*r+i))
            totals.append(total)
    return {'policy':name,'episodes':len(totals),'mean':statistics.fmean(totals),
            'sd':statistics.pstdev(totals) if len(totals)>1 else 0.,
            'solved':sum(1 for x in totals if x>0)}
