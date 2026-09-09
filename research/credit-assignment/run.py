"""Training harness for the panel task.

A re-implementation of `tcn.training.JointTrainer.episode` with the loss weights,
the discount and the supervision made configurable, because
`TrainConfig.__post_init__` rejects `prediction_weight <= 0` and `policy_weight <= 0`
and so cannot express a reward-only arm, a supervision-only arm, or the gamma = 0
myopic arm at all. That blocking defect is the one
`research/policy-learning/RESULTS.md` already reported; it is unchanged, and the
required diff is restated in `RESULTS.md`.

Everything else follows the stock trainer: one categorical over the action
templates masked by the generator's own menu, `tcn/policy.py:bind_action` for the
typed arguments, the shipped value baseline, Adam, grad-clip 5.
"""
from __future__ import annotations
import json,math,random,statistics,sys,time
from dataclasses import dataclass,field
import torch
ROOT='/home/brandonin/Documents/typed-crystallization-networks'
if ROOT not in sys.path:sys.path.insert(0,ROOT)
sys.path.insert(0,ROOT+'/research/credit-assignment')
from tcn.types import Value
from tcn.learning import SoftProgram,tensor
from tcn.policy import bind_action,action_inputs
from panel import Counter,TEMPLATES,BINDINGS,HORIZON
from program import panel_program

# Nodes whose relaxation must be sharp for the *executed action* to be the one the
# program computes: the two byte addresses, the byte comparison, and the two
# thresholds. None of them is a searched choice in the default scaffold.
SHARP=('byte','brandbyte','brand','prev_dial','slot_over')


@dataclass
class Cfg:
    episodes:int=800
    horizon:int=HORIZON
    seed:int=0
    lr:float=.05
    discount:float=.95
    w_actor:float=1.
    w_value:float=.5
    w_entropy:float=.02
    w_probe:float=0.
    batch:int=1
    perception:bool=False
    policy_state:bool=True
    slot_pool:bool=True
    train_choices:bool=True
    train_constants:bool=True
    index_offset:int=0
    eval_every:int=0
    eval_n:int=32
    seed_logits:dict|None=None
    sharp:float=.02


class Runner:
    def __init__(self,cfg:Cfg,counter:Counter):
        self.cfg=cfg;self.counter=counter
        torch.manual_seed(cfg.seed);torch.set_num_threads(1)
        program,registry=panel_program(perception=cfg.perception,policy_state=cfg.policy_state,
                                       slot_pool=cfg.slot_pool,seed_logits=cfg.seed_logits)
        self.program=program;self.registry=registry
        self.model=SoftProgram(program,registry)
        # Declared temperatures, stated because they are a hand-setting.
        # `relaxed`'s `index` is a softmax over byte positions and its `gt` is a
        # sigmoid, both at temperature 1 by default: at that width the relaxed
        # "byte at length-1" is a blend of three neighbouring bytes and no `eq`
        # against it can fire, so the program's *perception is wrong in the
        # relaxed forward* even when its choices are exactly right -- measured
        # here: the reference program scored 0.00/1 until these were set and
        # 1.00/1 after. Sharpening is what `ARCHITECTURE.md` section 5's anneal
        # does; setting it at the start on nodes whose choice is declared costs
        # no search, and `SHARP` names every node it touches.
        for name in SHARP:self.model.temperatures[name]=cfg.sharp
        params=[]
        if cfg.train_choices:params+=[p for p in self.model.choices if p.requires_grad]
        if cfg.train_constants:params+=list(self.model.constants.parameters())
        for p in self.model.parameters():p.requires_grad_(any(p is q for q in params))
        self.params=params
        self.optimizer=torch.optim.Adam(params,lr=cfg.lr) if params else None
        self.history=[];self.types=dict(program.inputs)

    def inputs(self,view,previous,executed):
        values={'terminal':tensor(view.observations['terminal'])}
        values['action']=torch.nn.functional.one_hot(torch.tensor(previous),len(TEMPLATES)).float()
        values.update({k:tensor(v) for k,v in action_inputs(BINDINGS,self.types,previous,executed).items()})
        return values

    def rollout(self,index,train=True,split='train'):
        c=self.cfg
        host=self.counter.create(c.index_offset+index if train else index,seed=c.seed,split=split,horizon=c.horizon)
        rows=[];previous=0;executed=None
        for t in range(c.horizon):
            if host.records[-1].done:break
            view=host.view()
            out,_=self.model(self.inputs(view,previous,executed))
            logits=out['policy'].flatten()
            allowed=torch.tensor([a.verb in view.available_actions for a in TEMPLATES],dtype=torch.bool)
            distribution=torch.distributions.Categorical(logits=logits.masked_fill(~allowed,float('-inf')))
            choice=distribution.sample() if train else logits.masked_fill(~allowed,float('-inf')).argmax()
            i=int(choice)
            action,argument_logp,argument_entropy=bind_action(TEMPLATES[i],i,BINDINGS,out,deterministic=not train)
            record=self.counter.step(host,action)
            reward=sum(v.decoded for v in record.reward_components.values())
            rows.append({'logp':distribution.log_prob(choice)+argument_logp,
                         'entropy':distribution.entropy()+argument_entropy,
                         'value':out['value'].reshape(()),'probe':out['probe'].flatten(),
                         'reward':float(reward),'verb':action.verb,
                         'target':(float(record.probes['answer'].decoded),
                                   float(record.probes['showing_task'].decoded))})
            previous=i;executed=action
            if record.done:break
        g=0.;returns=[]
        for row in reversed(rows):g=row['reward']+c.discount*g;returns.insert(0,g)
        return rows,returns,host

    def loss(self,rows,returns):
        c=self.cfg
        advantage=[torch.tensor(float(r))-row['value'].detach() for row,r in zip(rows,returns)]
        actor=torch.stack([-row['logp']*a for row,a in zip(rows,advantage)]).mean()
        value=torch.stack([(row['value']-float(r)).square() for row,r in zip(rows,returns)]).mean()
        entropy=torch.stack([row['entropy'] for row in rows]).mean()
        total=c.w_actor*actor+c.w_value*value-c.w_entropy*entropy
        probe=torch.stack([(row['probe']-torch.tensor(row['target'])).square().mean() for row in rows]).mean()
        if c.w_probe:total=total+c.w_probe*probe
        return total,{'actor':float(actor.detach()),'value':float(value.detach()),
                      'entropy':float(entropy.detach()),'probe':float(probe.detach())}

    def train(self,log_every=50,evaluate_at=()):
        c=self.cfg;i=0;curve=[];window=[]
        while i<c.episodes:
            n=min(c.batch,c.episodes-i)
            self.optimizer.zero_grad()
            terms=[];rewards=[]
            for k in range(n):
                rows,returns,_=self.rollout(i+k)
                total,info=self.loss(rows,returns)
                (total/n).backward()
                terms.append(info);rewards.append(sum(r['reward'] for r in rows));window.append(rewards[-1])
            gn=float(torch.nn.utils.clip_grad_norm_(self.params,5.))
            self.optimizer.step();i+=n
            if log_every and i%log_every<n:
                self.history.append({'episode':i,'grad_norm':gn,'train_return':statistics.fmean(window),
                                     'reward_rate':sum(1 for x in window if x>0)/len(window),
                                     'selection':self.candidate_names(),
                                     **{k:statistics.fmean(t[k] for t in terms) for k in terms[0]}})
                window=[]
            if i in evaluate_at:
                curve.append({'episode':i,'eval':self.evaluate(c.eval_n)})
        return curve

    def evaluate(self,n=32,start=10000,split='test'):
        totals=[];verbs=[]
        with torch.no_grad():
            for k in range(n):
                rows,_,_=self.rollout(start+k,train=False,split=split)
                totals.append(sum(r['reward'] for r in rows));verbs.append([r['verb'] for r in rows])
        return {'mean':statistics.fmean(totals),'solved':sum(1 for x in totals if x>0),'n':n,
                'modal_trace':statistics.mode([tuple(v) for v in verbs])}

    def selections(self):
        return {k:v for k,v in self.model.selections().items()}

    def report(self):
        constants={k:float(v.detach()) for k,v in self.model.constants.items()}
        return {'selections':self.selections(),
                'logits':{name:[round(constants[f'{name}{i}'],3) for i in range(4)]
                          for name in ('idle','found','dialled') if f'{name}0' in constants},
                'candidate':self.candidate_names()}

    def candidate_names(self):
        out={}
        for node,p in zip(self.program.nodes,self.model.choices):
            if len(node.candidates)>1:
                i=int(torch.argmax(p))
                c=node.candidates[i]
                out[node.name]=f'{c.operator.name}({",".join(c.sources)})'
        return out


def freeze_perception(runner):
    """Crystallize every searched perception/transform node at its argmax."""
    frozen=[]
    for node in runner.program.nodes:
        if len(node.candidates)>1 and node.region in {'perception','transform'} and node.name not in runner.model.frozen:
            runner.model.freeze(node.name);frozen.append(node.name)
    runner.params=[p for p in runner.model.choices if p.requires_grad]+list(runner.model.constants.parameters())
    for p in runner.model.parameters():p.requires_grad_(False)
    for p in runner.params:p.requires_grad_(True)
    runner.optimizer=torch.optim.Adam(runner.params,lr=runner.cfg.lr)
    return frozen
