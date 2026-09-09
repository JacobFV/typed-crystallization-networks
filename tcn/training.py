"""Generic simultaneous prediction, probe, and policy optimization on typed episodes."""
from dataclasses import dataclass,asdict
from pathlib import Path
import json
import random
import torch
from .types import Value
from .generation import Host,Action,source_fingerprint,load_generator
from .learning import SoftProgram,tensor
from .policy import ActionBinding,bind_action,action_inputs,parameter_width

@dataclass(frozen=True)
class Target:
    channel: str
    key: str
    horizon: int = 1
    scale: float = 1.
    loss: str = 'mse'
    def __post_init__(self):
        if self.loss not in {'mse','bernoulli','gaussian'}:raise ValueError('invalid target likelihood')
        if self.channel not in {'probes','latent_states','observations'} or self.horizon<0 or self.scale<=0:raise ValueError('invalid prediction target')
    def value(self,record):
        v=getattr(record,self.channel)[self.key]
        return v.value if self.channel=='observations' else v

@dataclass
class TrainConfig:
    generator: str
    observations: tuple[str,...]
    targets: tuple[Target,...]
    action_templates: tuple[Action,...]
    generator_config: dict
    objectives: tuple[dict,...] = ({},)
    episodes: int = 64
    horizon: int = 8
    dt: float = 1.
    lr: float = .01
    discount: float = .95
    prediction_weight: float = 1.
    probe_weight: float = 0.
    policy_weight: float = 1.
    value_weight: float = .5
    entropy_weight: float = .01
    mdl_weight: float = 0.
    crystal_weight: float = .001
    seed: int = 0
    action_bindings: tuple[ActionBinding,...] = ()
    def __post_init__(self):
        if not self.objectives or not self.action_templates:raise ValueError('goals and actions must be nonempty')
        if self.prediction_weight<=0 or self.policy_weight<=0:raise ValueError('integrated training requires prediction and policy objectives')
        if len({(b.template,b.argument) for b in self.action_bindings})!=len(self.action_bindings):raise ValueError('duplicate action binding')
        if any(b.template>=len(self.action_templates) for b in self.action_bindings):raise ValueError('action binding index out of range')
        if self.episodes<1 or self.horizon<1 or self.dt<=0:raise ValueError('invalid training budget')
    def to_dict(self):
        d=asdict(self);d['action_templates']=[a.to_dict() for a in self.action_templates];d['action_bindings']=[b.to_dict() for b in self.action_bindings];return d
    @classmethod
    def from_dict(cls,d):
        d=dict(d);d['targets']=tuple(Target(**t) for t in d['targets']);d['action_templates']=tuple(Action.from_dict(a) for a in d['action_templates']);d['observations']=tuple(d['observations']);d['objectives']=tuple(d['objectives']);d['action_bindings']=tuple(ActionBinding.from_dict(x) for x in d.get('action_bindings',()));return cls(**d)

def target_loss(target,pred,truth):
    if target.loss=='mse':return (pred-truth).square().mean()
    if target.loss=='bernoulli':return torch.nn.functional.binary_cross_entropy_with_logits(pred,truth)
    mean,logstd=pred.chunk(2);logstd=logstd.clamp(-10,5)
    return (.5*((truth-mean)*(-logstd).exp()).square()+logstd+.5*__import__('math').log(2*__import__('math').pi)).mean()

class JointTrainer:
    def __init__(self,model,config):
        self.model=model;self.config=config
        self.optimizer=torch.optim.Adam(list(model.parameters()),lr=config.lr)
        self.history=[];self.completed=0
        outputs={k:model.program.port_types()[v] for k,v in model.program.outputs}
        gen=load_generator(config.generator);gen.configure(config.generator_config)
        for b in config.action_bindings:
            if b.argument!='__target__' and gen.action_schema[config.action_templates[b.template].verb].get(b.argument)!=b.type:raise TypeError('binding/action signature mismatch')
            if outputs[b.output] is None or outputs[b.output].width!=parameter_width(b.type):raise TypeError('binding/output signature mismatch')
        action_inputs(config.action_bindings,dict(model.program.inputs))
    def inputs(self,view,action,executed=None):
        types=dict(self.model.program.inputs)
        for k in self.config.observations:
            if view.observations[k].type!=types[k]:raise TypeError('observation signature mismatch: '+k)
        device=next(self.model.parameters()).device
        values={k:tensor(view.observations[k],device) for k in self.config.observations}
        values.update({k:tensor(v,device) for k,v in action_inputs(self.config.action_bindings,types,action,executed).items()})
        if 'action' in dict(self.model.program.inputs):values['action']=torch.nn.functional.one_hot(torch.tensor(action,device=device),len(self.config.action_templates)).float()
        if 'dt' in dict(self.model.program.inputs):values['dt']=torch.tensor([self.config.dt],device=device)
        return values
    def episode(self,index,train=True,split='train',loss_only=False,regularized=True):
        # `regularized=False` drops the two terms that attach to the architecture
        # parameters directly rather than through the graph -- the discreteness
        # term over choice distributions and the description-size term over
        # candidate costs. It is the unregularized task objective the
        # crystallizer's connectivity guard must probe: with those terms present
        # every unfrozen logit stays reachable in the autograd graph whether or
        # not any task signal reaches it. The policy entropy bonus is retained,
        # because it flows through the program's own policy output and so is a
        # genuine gradient path through the graph.
        c=self.config;goal=c.objectives[index%len(c.objectives)]
        host=Host.create(c.generator,seed=c.seed,index=index,split=split,configuration=c.generator_config|{'horizon':c.horizon},objective=goal)
        memory=None;previous=0;executed=None;rows=[]
        for t in range(c.horizon):
            view=host.view();inputs=self.inputs(view,previous,executed)
            out,_,trace=self.model(inputs,memory,return_trace=True)
            logits=out['policy'].flatten()
            if len(logits)!=len(c.action_templates):raise ValueError('policy width/action library mismatch')
            allowed=torch.tensor([a.verb in view.available_actions for a in c.action_templates],dtype=torch.bool,device=logits.device)
            if not allowed.any():raise ValueError('no visible action candidate')
            distribution=torch.distributions.Categorical(logits=logits.masked_fill(~allowed,float('-inf')))
            choice=distribution.sample() if train else logits.masked_fill(~allowed,float('-inf')).argmax()
            i=int(choice);action,argument_logp,argument_entropy=bind_action(c.action_templates[i],i,c.action_bindings,out,deterministic=not train)
            pred_inputs=self.inputs(view,i,action)
            predicted,newmemory,_=self.model(pred_inputs,memory,return_trace=True)
            record=host.step((action,),c.dt)
            reward=sum(v.decoded for v in record.reward_components.values())
            rows.append({'logp':distribution.log_prob(choice)+argument_logp,'entropy':distribution.entropy()+argument_entropy,'value':out['value'].reshape(()),'prediction':predicted['prediction'].flatten(),'probe':predicted.get('probe'),'reward':float(reward)})
            memory=newmemory;previous=i;executed=action
            if record.done:break
        returns=[];g=0.
        for row in reversed(rows):g=row['reward']+c.discount*g;returns.insert(0,g)
        pred_losses=[];probe_losses=[]
        for t,row in enumerate(rows):
            offset=0
            for target in c.targets:
                at=t+target.horizon;width=target.value(host.records[0]).type.width
                size=width*(2 if target.loss=='gaussian' else 1)
                pred=row['prediction'][offset:offset+size];offset+=size
                if at<len(host.records):
                    truth=tensor(target.value(host.records[at]),pred.device)/target.scale
                    pred_losses.append(target_loss(target,pred,truth))
            if offset!=len(row['prediction']):raise ValueError('prediction output/target signature mismatch')
            if row['probe'] is not None:
                truth=torch.cat([tensor(target.value(host.records[t]),row['probe'].device)/target.scale for target in c.targets])
                probe_losses.append((row['probe'].flatten()-truth).square().mean())
        if not pred_losses:raise ValueError('no future target within rollout')
        prediction=torch.stack(pred_losses).mean();probe=torch.stack(probe_losses).mean() if probe_losses else prediction*0
        actor=torch.stack([-r['logp']*(ret-r['value'].detach()) for r,ret in zip(rows,returns)]).mean()
        value=torch.stack([(r['value']-ret).square() for r,ret in zip(rows,returns)]).mean()
        entropy=torch.stack([r['entropy'] for r in rows]).mean()
        loss=c.prediction_weight*prediction+c.probe_weight*probe+c.policy_weight*actor+c.value_weight*value-c.entropy_weight*entropy
        if regularized:loss=loss+c.mdl_weight*self.model.complexity()+c.crystal_weight*self.model.entropy()
        if loss_only:return loss
        metrics={'episode':index,'return':sum(r['reward'] for r in rows),'prediction_loss':float(prediction.detach()),'probe_loss':float(probe.detach()),'policy_loss':float(actor.detach()),'value_loss':float(value.detach()),'loss':float(loss.detach()),'goal':goal,'split':split}
        if train:
            active=[p for p in self.model.parameters() if p.requires_grad]
            pg=torch.autograd.grad(prediction,active,retain_graph=True,allow_unused=True) if prediction.requires_grad else [None]*len(active)
            rg=torch.autograd.grad(actor,active,retain_graph=True,allow_unused=True) if actor.requires_grad else [None]*len(active)
            shared=[(a.flatten(),b.flatten()) for a,b in zip(pg,rg) if a is not None and b is not None]
            if shared:
                a=torch.cat([a for a,b in shared]);b=torch.cat([b for a,b in shared]);den=a.norm()*b.norm()
                metrics['prediction_policy_gradient_cosine']=float((a@b/den).detach()) if den>0 else 0.
            self.optimizer.zero_grad();loss.backward()
            metrics['gradient_norm']=float(torch.nn.utils.clip_grad_norm_(self.model.parameters(),5.))
            self.optimizer.step()
        return metrics,host
    def run(self,outdir=None):
        torch.set_num_threads(1)
        if self.completed==0:torch.manual_seed(self.config.seed)
        for i in range(self.completed,self.config.episodes):
            metrics,_=self.episode(i);self.history.append(metrics);self.completed=i+1
            if outdir is not None and (self.completed%8==0 or self.completed==self.config.episodes):self.save(outdir)
        return self.history
    def save(self,outdir):
        p=Path(outdir);p.mkdir(parents=True,exist_ok=True)
        torch.save({'format':'tcn.training/1','source':source_fingerprint(),'modules':[m.to_dict() for m in self.model.registry.modules.values()],'requires_grad':{k:p.requires_grad for k,p in self.model.named_parameters()},'trials':self.model.trials,'quantization':{k:(t.to_dict(),pressure) for k,(t,pressure) in self.model.quantization.items()},'program':self.model.program.to_dict(),'state':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),'temperatures':self.model.temperatures,'frozen':self.model.frozen,'config':self.config.to_dict(),'completed':self.completed,'history':self.history,'rng':torch.get_rng_state()},p/'checkpoint.pt')
        (p/'metrics.json').write_text(json.dumps(self.history,indent=2))
    @classmethod
    def load(cls,path,registry=None):
        from .graph import Program
        d=torch.load(path,weights_only=True,map_location='cpu')
        if d['format']!='tcn.training/1':raise ValueError('checkpoint version mismatch')
        if d['source']!=source_fingerprint():raise ValueError('checkpoint source revision mismatch')
        from .operators import Registry
        from .types import Type
        registry=registry or Registry()
        for spec in d['modules']:registry.register_module(Program.from_dict(spec,registry))
        m=SoftProgram(Program.from_dict(d['program'],registry),registry);m.load_state_dict(d['state']);m.temperatures=d['temperatures'];m.frozen=d['frozen']
        for k,p in m.named_parameters():p.requires_grad_(d['requires_grad'][k])
        m.trials=d['trials'];m.quantization={k:(Type.from_dict(v[0]),v[1]) for k,v in d['quantization'].items()}
        t=cls(m,TrainConfig.from_dict(d['config']));t.optimizer.load_state_dict(d['optimizer']);t.completed=d['completed'];t.history=d['history'];torch.set_rng_state(d['rng']);return t
