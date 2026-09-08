"""Typed action distributions over the same four carriers; no domain action heads."""
from dataclasses import dataclass
import math
from .types import Type,Value,decode
from .generation import Action,read_text

@dataclass(frozen=True)
class ActionBinding:
    template: int
    argument: str
    type: Type
    output: str
    bounds: tuple[float,float] = (-1.,1.)
    def __post_init__(self):
        if self.template<0 or not self.argument or not self.output or not self.bounds[1]>self.bounds[0]:raise ValueError('invalid action binding')
    @property
    def input_port(self):return f'action.{self.template}.{self.argument}'
    def to_dict(self):return {'template':self.template,'argument':self.argument,'type':self.type.to_dict(),'output':self.output,'bounds':list(self.bounds)}
    @classmethod
    def from_dict(cls,d):return cls(d['template'],d['argument'],Type.from_dict(d['type']),d['output'],tuple(d.get('bounds',(-1.,1.))))

def parameter_width(t):
    if t.kind=='bool':return 1
    if t.kind=='int':return t.bits if t.role in {'category','symbol'} else 2
    if t.kind=='tuple':return sum(parameter_width(x) for x in t.items)
    return t.capacity*(1+parameter_width(t.items[0]))

def numeric_bounds(t,default):
    if t.bounds is not None:return t.bounds
    if t.encoding.kind=='float':return default
    scale=t.encoding.scale if t.encoding.kind=='fixed' else 1.
    return ((-2**(t.bits-1) if t.encoding.signed else 0)/scale,(2**(t.bits-int(t.encoding.signed))-1)/scale)

def sample_typed(t,parameters,bounds=(-1.,1.),deterministic=False):
    """Score-function sampling; exact encoding follows a bounded latent draw."""
    import torch
    params=parameters.flatten();offset=0;terms=[];entropies=[]
    if len(params)!=parameter_width(t):raise ValueError('typed policy parameter width mismatch')
    def bit():
        nonlocal offset
        dist=torch.distributions.Bernoulli(logits=params[offset]);offset+=1
        v=(dist.probs>=.5).float() if deterministic else dist.sample()
        terms.append(dist.log_prob(v));entropies.append(dist.entropy());return bool(v.item())
    def walk(t):
        nonlocal offset
        if t.kind=='bool':return bit()
        if t.kind=='int' and t.role in {'category','symbol'}:
            raw=sum(int(bit())<<i for i in range(t.bits));return decode(t,raw)
        if t.kind=='int':
            mean=params[offset];logstd=params[offset+1].clamp(-5,2);offset+=2
            dist=torch.distributions.Normal(mean,logstd.exp());u=mean.detach() if deterministic else dist.sample()
            lo,hi=numeric_bounds(t,bounds)
            if not hi>lo:raise ValueError('nonempty numeric policy range required')
            z=u.tanh();x=lo+(z+1)*.5*(hi-lo)
            terms.append(dist.log_prob(u)-torch.log(1-z*z+1e-6)-math.log((hi-lo)/2));entropies.append(dist.entropy())
            x=float(x.item());return max(int(lo),min(int(hi),round(x))) if t.encoding.kind=='integer' else max(lo,min(hi,x))
        if t.kind=='tuple':return tuple(walk(x) for x in t.items)
        values=[]
        for _ in range(t.capacity):
            present=bit()
            if present:values.append(walk(t.items[0]))
            else:offset+=parameter_width(t.items[0])
        return frozenset(values)
    value=Value.of(t,walk(t));zero=params.sum()*0
    return value,sum(terms,zero),sum(entropies,zero)

def sample_exact(t,parameters,rng,bounds=(-1.,1.),deterministic=False):
    """Standard-library sampler for frozen execution; same latent distribution."""
    params=iter(parameters)
    def bit():
        x=float(next(params));p=1/(1+math.exp(-max(-700,min(700,x))))
        return p>=.5 if deterministic else rng.random()<p
    def walk(t):
        if t.kind=='bool':return bit()
        if t.kind=='int' and t.role in {'category','symbol'}:return decode(t,sum(int(bit())<<i for i in range(t.bits)))
        if t.kind=='int':
            mean=float(next(params));std=math.exp(max(-5,min(2,float(next(params)))))
            lo,hi=numeric_bounds(t,bounds);u=mean if deterministic else rng.gauss(mean,std);x=lo+(math.tanh(u)+1)*.5*(hi-lo)
            return max(int(lo),min(int(hi),round(x))) if t.encoding.kind=='integer' else max(lo,min(hi,x))
        if t.kind=='tuple':return tuple(walk(x) for x in t.items)
        out=[]
        for _ in range(t.capacity):
            if bit():out.append(walk(t.items[0]))
            else:
                for _ in range(parameter_width(t.items[0])):next(params)
        return frozenset(out)
    v=Value.of(t,walk(t))
    if next(params,None) is not None:raise ValueError('unused policy parameters')
    return v

def bind_action(template,index,bindings,outputs,training=True,rng=None,deterministic=False):
    args=dict(template.arguments);target=template.target;scores=[];entropies=[]
    for binding in bindings:
        if binding.template!=index:continue
        if training:
            value,score,entropy=sample_typed(binding.type,outputs[binding.output],binding.bounds,deterministic);scores.append(score);entropies.append(entropy)
        else:value=sample_exact(binding.type,outputs[binding.output].flat(),rng,binding.bounds,deterministic)
        if binding.argument=='__target__':target=read_text(value)
        else:args[binding.argument]=value
    return Action(template.verb,target,tuple(args.items()),template.actor),sum(scores),sum(entropies)


def action_inputs(bindings,types,index=None,action=None):
    """Declared argument ports carry executed values; inactive templates use defaults."""
    result={}
    for b in bindings:
        if b.input_port not in types:raise ValueError('missing executed-action input '+b.input_port)
        if types[b.input_port]!=b.type:raise TypeError('executed-action input type mismatch')
        value=None
        if index==b.template and action is not None:
            if b.argument=='__target__':
                from .generation import text_value
                value=text_value(action.target,len(b.type.items[1].items))
            else:value=dict(action.arguments)[b.argument]
        if value is None:value=sample_exact(b.type,[0.]*parameter_width(b.type),None,b.bounds,True)
        result[b.input_port]=value
    return result
