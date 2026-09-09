"""Type-checked soft choices; exact hard-forward trials; frozen gradient boundaries."""
from __future__ import annotations
import json
import math
from dataclasses import replace
import torch
from torch import nn
from .types import Value, decode
from .operators import Registry, BINARY, UNARY, COMPARE, CONVERSIONS

def tensor(value,device=None): return torch.tensor(value.flat(),dtype=torch.float32,device=device)
def ste(soft,hard): return soft+(hard-soft).detach()

def exact_tensor(registry,op,xs):
    shape=torch.broadcast_shapes(*(x.shape[:-1] for x in xs)) if xs else ()
    batch=math.prod(shape) if shape else 1
    flat=[x.detach().expand(*shape,x.shape[-1]).reshape(batch,-1).cpu().tolist() for x in xs]
    # Exact execution is a pure function of the argument row, and a typed row over
    # finite carriers repeats: a 3-input Boolean module has at most 8 distinct
    # argument rows whatever the batch, so a 64-row truth table holds 8 real calls
    # and 56 repeats. Evaluate each distinct row once. This is memoization of a
    # pure function within one call -- no new semantics, and the returned tensor is
    # bit-identical. Rows that never repeat (wide integer or float carriers) pay
    # only the hashing of the key, measured at 4-13%.
    #
    # The key is the row's decoded values. Two distinct float rows can compare
    # equal only through signed zero (`-0.0 == 0.0`), and that difference is
    # observable in exact execution (`atan2`), so a call whose arguments contain a
    # negative zero skips the cache rather than risking a wrong reuse. NaN keys
    # never match and simply miss, which is safe.
    cache=None if any(x.dtype.is_floating_point and bool((torch.signbit(x.detach())&(x.detach()==0)).any()) for x in xs) else {}
    vals=[]
    for i in range(batch):
        if cache is None:
            vals.append(registry.exact(op,[Value.unflat(t,x[i]) for t,x in zip(op.inputs,flat)]).flat()); continue
        key=tuple(tuple(x[i]) for x in flat)
        got=cache.get(key)
        if got is None:
            args=[Value.unflat(t,x[i]) for t,x in zip(op.inputs,flat)]
            got=cache[key]=registry.exact(op,args).flat()
        vals.append(got)
    device=xs[0].device if xs else None
    return torch.tensor(vals,dtype=torch.float32,device=device).reshape(*shape,op.output.width)

def carrier_temperature(t):
    """`eq`'s surrogate scale derived from a declared carrier, not tuned.

    `eq` relaxes to `exp(-(a-b)^2/tau)`. At tau=1 that is **exactly 0.0** in
    float32 once `|a-b| >= 11` (1.6e-28 is still representable at 8), so on a
    carrier that can express differences of `2^bits` the surrogate -- and its
    gradient -- is dead for all but near-equal values. `int[8]` image bytes sit
    25.6 apart at a uniform mixture and read 3.2e-31, i.e. zero.

    `tau = 2^bits` is the smallest scale on which a full-carrier disagreement is
    still representable: at the measured failing distance it lifts the surrogate
    from 0.0 to 7.7e-02. It is read off the declared type, so a `bool` carrier
    returns 1. and its relaxation is unchanged. Composite carriers take the
    widest leaf, since `eq` sums the squared difference over the whole width.
    """
    leaves=[]
    def walk(x):
        if x.kind=="int" and x.encoding.kind=="integer": leaves.append(float(2**x.bits))
        elif x.items:
            for y in x.items: walk(y)
        else: leaves.append(1.)
    walk(t)
    return max(leaves) if leaves else 1.

def relaxed(registry,op,xs,temperature=1.,carrier_scaled=False):
    """`carrier_scaled` widens `eq`'s surrogate to its declared carrier.

    It defaults to False so every shipped relaxation is bit-identical. Turning
    it on is only ever correct together with a *separate* surrogate temperature
    (see `SoftProgram.surrogate_scale`): one temperature per node divides both
    the candidate softmax and the operator relaxation, so widening a surrogate
    through `temperature` also flattens that node's choice distribution.
    """
    n=op.name; p=dict(op.parameters)
    if op.gradient=="none": return exact_tensor(registry,op,xs)
    a=xs[0] if xs else None; b=xs[1] if len(xs)>1 else None
    if n=="identity": return a
    if n=="not": return 1-a
    if n in {"and","or","xor","nand","nor","xnor"}:
        base={"and":a*b,"or":a+b-a*b,"xor":a+b-2*a*b}
        return base[n] if n in base else 1-base[{"nand":"and","nor":"or","xnor":"xor"}[n]]
    if n.startswith("truth_"):
        k=int(n[6:]); terms=((1-a)*(1-b),(1-a)*b,a*(1-b),a*b)
        return sum(v*((k>>i)&1) for i,v in enumerate(terms))
    if n=="mux": return a*xs[1]+(1-a)*xs[2]
    if n in COMPARE:
        if n=="eq":
            tau=temperature*(carrier_temperature(op.inputs[0]) if carrier_scaled else 1.)
            return torch.exp(-((a-b)**2).sum(-1,keepdim=True)/tau)
        d=b-a if n in {"lt","le"} else a-b
        return torch.sigmoid(d/temperature)
    if n in BINARY | UNARY:
        if n in {"div","mod","idiv"} and torch.any(b==0): raise ValueError("invalid denominator in relaxed candidate")
        if n=="log" and torch.any(a<=0): raise ValueError("invalid relaxed log domain")
        if n=="sqrt" and torch.any(a<0): raise ValueError("invalid relaxed sqrt domain")
        funcs={"add":lambda:a+b,"sub":lambda:a-b,"mul":lambda:a*b,"div":lambda:a/b,"pow":lambda:a**b,"mod":lambda:torch.remainder(a,b),"idiv":lambda:ste(a/b,torch.floor(a/b)),"min":lambda:torch.minimum(a,b),"max":lambda:torch.maximum(a,b),"shl":lambda:a*2**b,"shr":lambda:a/2**b,"atan2":lambda:torch.atan2(a,b),"neg":lambda:-a,"abs":lambda:a.abs(),"exp":lambda:a.exp(),"log":lambda:a.log(),"sin":lambda:a.sin(),"cos":lambda:a.cos(),"sqrt":lambda:a.sqrt()}
        y=funcs[n]()
        if not torch.all(torch.isfinite(y)): raise ValueError("nonfinite relaxed result")
        return y
    if n=="tuple": return torch.cat(xs,dim=-1) if xs else torch.empty(0)
    if n=="project":
        widths=[t.width for t in op.inputs[0].items]; i=p["index"]; start=sum(widths[:i])
        return a[...,start:start+widths[i]]
    if n=="index":
        width=op.output.width; count=len(op.inputs[0].items)
        weights=torch.softmax(-(b-torch.arange(count,device=a.device))**2/temperature,dim=-1)
        return (a.reshape(*a.shape[:-1],count,width)*weights.unsqueeze(-1)).sum(-2)
    if n in {"sum","mean","reduce_min","reduce_max","count"}:
        if n=="count":
            return a.sum(-1,keepdim=True) if op.inputs[0].kind=="set" else torch.full((*a.shape[:-1],1),float(len(op.inputs[0].items)),device=a.device)
        return {"sum":lambda:a.sum(-1,keepdim=True),"mean":lambda:a.mean(-1,keepdim=True),"reduce_min":lambda:a.min(-1,keepdim=True).values,"reduce_max":lambda:a.max(-1,keepdim=True).values}[n]()
    if n=="union": return a+b-a*b
    if n=="intersection": return a*b
    if n in {"delay","difference","accumulation"}:
        return torch.cat((b,a),-1) if n=="delay" else (torch.cat((a-b,a),-1) if n=="difference" else torch.cat((a+b,a+b),-1))
    if n=="fft":
        z=torch.fft.fft(a,dim=-1); return torch.view_as_real(z).flatten(-2)
    if n=="ifft":
        z=torch.view_as_complex(a.reshape(*a.shape[:-1],-1,2).contiguous())
        return torch.view_as_real(torch.fft.ifft(z,dim=-1)).flatten(-2)
    if n=="fourier_basis": return torch.cat((a.cos(),a.sin()),-1)
    if n in CONVERSIONS:
        if op.output.kind=="bool": return torch.sigmoid((a-p.get("threshold",.5))/temperature)
        return a
    raise NotImplementedError(f"missing declared relaxation for {n}")

class SoftProgram(nn.Module):
    def __init__(self,program,registry=None):
        super().__init__(); self.registry=registry or Registry(); self.program=program.validate(self.registry)
        self.choices=nn.ParameterList([nn.Parameter(torch.zeros(len(n.candidates)),requires_grad=n.selected is None) for n in program.nodes])
        self.constants=nn.ParameterDict({k:nn.Parameter(tensor(dict(program.constants)[k])) for k in program.trainable_constants})
        self.temperatures={n.name:1. for n in program.nodes}
        # D2, separated. One temperature per node used to divide BOTH the
        # candidate softmax and the operator's own relaxation, so a surrogate
        # could not be widened without flattening that node's choice
        # distribution at the same time. `surrogate_scale` is the second axis:
        # the relaxation sees `temperatures[n] * surrogate_scale[n]` while the
        # softmax, the entropy and the description cost keep seeing
        # `temperatures[n]` alone. It defaults to 1. everywhere, so every
        # shipped run is bit-identical, and the crystallizer's anneal still
        # sharpens surrogates proportionally because it scales the shared term.
        self.surrogate_scale={n.name:1. for n in program.nodes}
        self.carrier_scaled=False
        self.frozen={n.name:n.selected for n in program.nodes if n.selected is not None}
        self.trials={}
        self.quantization={}
    def forward(self,inputs,state=None,return_trace=False):
        if set(inputs)!=set(dict(self.program.inputs)): raise ValueError("input ports mismatch")
        values={k:(tensor(v,self.choices[0].device if len(self.choices) else None) if isinstance(v,Value) else v) for k,v in inputs.items()}
        for k,t in self.program.inputs:
            if values[k].shape[-1]!=t.width: raise TypeError("input width mismatch")
            if isinstance(inputs[k],Value) and inputs[k].type!=t: raise TypeError("input type mismatch")
        ref=next(iter(values.values()),torch.zeros(1))
        for k,v in self.program.constants: values[k]=self.constants[k] if k in self.constants else tensor(v,ref.device)
        for k,v,_ in self.program.state: values[k]=tensor(v,ref.device) if state is None else state[k]
        for n,logits in zip(self.program.nodes,self.choices):
            if n.name in self.frozen:
                c=n.candidates[self.frozen[n.name]]
                values[n.name]=exact_tensor(self.registry,c.operator,[values[s] for s in c.sources]).detach()
                continue
            tau=self.temperatures[n.name]
            sur=tau*self.surrogate_scale[n.name]
            ys=[relaxed(self.registry,c.operator,[values[s] for s in c.sources],sur,self.carrier_scaled) for c in n.candidates]
            ys=torch.broadcast_tensors(*ys)
            weights=torch.softmax(logits/tau,0)
            y=sum(w*z for w,z in zip(weights,ys))
            if n.name in self.trials:
                c=n.candidates[self.trials[n.name]]
                y=ste(y,exact_tensor(self.registry,c.operator,[values[s] for s in c.sources]))
            if n.name in self.quantization:
                qtype,pressure=self.quantization[n.name]
                if qtype!=n.output or not 0<=pressure<=1:raise ValueError("precision relaxation must target the declared node encoding")
                hard=exact_tensor(self.registry,self.registry.resolve("quantize",(n.output,),qtype),[y])
                y=y+pressure*(hard-y).detach()
            if not torch.all(torch.isfinite(y)): raise ValueError("nonfinite graph output")
            values[n.name]=y
        out={k:values[v] for k,v in self.program.outputs}; newstate={k:values[u] for k,_,u in self.program.state}
        return (out,newstate,values) if return_trace else (out,newstate)
    def entropy(self):
        terms=[]
        for n,p in zip(self.program.nodes,self.choices):
            if n.name not in self.frozen:
                q=torch.softmax(p/self.temperatures[n.name],0); terms.append(-(q*q.clamp_min(1e-12).log()).sum())
        return sum(terms,torch.zeros((),device=self.choices[0].device if len(self.choices) else None))
    def complexity(self):
        terms=[]
        for n,p in zip(self.program.nodes,self.choices):
            costs=torch.tensor([c.operator.cost for c in n.candidates],device=p.device)
            terms.append((torch.softmax(p/self.temperatures[n.name],0)*costs).sum())
        return sum(terms,torch.zeros((),device=self.choices[0].device if len(self.choices) else None))
    def distributions(self):
        """Per-node choice distribution: the softmax, or one-hot where hardened."""
        out=[]
        for n,p in zip(self.program.nodes,self.choices):
            if n.name in self.frozen:
                q=torch.zeros(len(n.candidates),device=p.device); q[self.frozen[n.name]]=1.
            else: q=torch.softmax(p/self.temperatures[n.name],0)
            out.append(q)
        return out
    def _description_tables(self):
        """Static per-candidate description sizes, source uses, and module closures.

        `Program.description_bits` is the byte length of the serialized program
        times eight, plus each distinct frozen module definition once. That
        decomposes exactly: the node list contributes one serialized node per
        live node plus its two separator characters, and everything else is a
        header that does not depend on which candidates are selected.
        """
        tables=getattr(self,'_desc_tables',None)
        if tables is not None: return tables
        registry=self.registry
        def referenced(op):
            names=[op.name,dict(op.parameters).get('module','')]
            return [n for n in names if isinstance(n,str) and n.startswith('module:')]
        closures={}
        def closure(name):
            if name in closures: return closures[name]
            seen=set(); stack=[name]
            while stack:
                m=stack.pop()
                if m in seen: continue
                seen.add(m)
                for node in registry.modules[m].nodes:
                    for c in node.candidates: stack+=referenced(c.operator)
            closures[name]=seen; return seen
        bits=[]; uses=[]; module_uses={}
        for i,n in enumerate(self.program.nodes):
            envelope=len(json.dumps({"name":n.name,"output":n.output.to_dict(),"candidates":[],"region":n.region,"depth":n.depth,"selected":0},sort_keys=True))
            bits.append(torch.tensor([8.*(envelope+len(json.dumps(c.to_dict(),sort_keys=True))+2) for c in n.candidates]))
            u={}
            for j,c in enumerate(n.candidates):
                for s in set(c.sources): u.setdefault(s,[]).append(j)
                for m in referenced(c.operator):
                    for name in closure(m): module_uses.setdefault(name,{}).setdefault(i,set()).add(j)
            uses.append(u)
        module_uses={m:{i:sorted(j) for i,j in per.items()} for m,per in module_uses.items()}
        module_bits={m:float(registry.modules[m].description_bits()) for m in module_uses}
        tables={'bits':bits,'uses':uses,'module_uses':module_uses,'module_bits':module_bits}
        self._desc_tables=tables; return tables
    def _description_header(self):
        """Bits of the serialized program that do not depend on the selection."""
        constants=tuple((k,Value.unflat(v.type,self.constants[k].detach().cpu().tolist()) if k in self.constants else v) for k,v in self.program.constants)
        empty=replace(self.program,nodes=(),constants=constants,trainable_constants=(),version=self.program.version+1)
        return 8.*(len(json.dumps(empty.to_dict(),sort_keys=True))-2)
    def description_cost(self):
        """Expected description length, in bits, of the pruned hardened program.

        This is the term ARCHITECTURE section 8 calls `L_program_description`, and
        it measures what `Program.description_bits` measures: the serialized
        program plus each distinct frozen module definition charged **once**,
        with call sites charged individually. It is not `complexity()`: that is a
        softmax-weighted sum of `operator.cost`, which is execution cost, and a
        module call's execution cost is at parity with its inlined body.

        Two expectations are taken under the current factorized choice
        distribution. A node is charged only when it is live, where liveness
        propagates backwards from the outputs and the state updates; a module
        definition is charged once, when *some* live node calls it. Both are
        products of independent per-node choice probabilities, so this is a
        mean-field relaxation: it ignores correlations between nodes while the
        distribution is diffuse, and at any one-hot distribution it is **exactly**
        `pruned().harden(selections).description_bits(registry)`.

        Bits, so the natural weight against a probe loss is around 1e-5.
        """
        tables=self._description_tables(); q=self.distributions()
        nodes=self.program.nodes; device=self.choices[0].device if len(self.choices) else None
        sinks={v for _,v in self.program.outputs}|{u for _,_,u in self.program.state}
        one=torch.ones((),device=device); live=[one]*len(nodes)
        for i in range(len(nodes)-1,-1,-1):
            if nodes[i].name in sinks: continue
            dead=one
            for k in range(i+1,len(nodes)):
                idx=tables['uses'][k].get(nodes[i].name)
                if idx: dead=dead*(1-live[k]*q[k][idx].sum())
            live[i]=1-dead
        total=torch.as_tensor(self._description_header(),device=device)
        for i,n in enumerate(nodes):
            total=total+live[i]*(q[i]*tables['bits'][i].to(device)).sum()
        for m,per_node in tables['module_uses'].items():
            dead=one
            for i,idx in per_node.items(): dead=dead*(1-live[i]*q[i][idx].sum())
            total=total+tables['module_bits'][m]*(1-dead)
        return total
    def scale_surrogates(self):
        """Turn on the carrier-derived `eq` surrogate width, without touching choices.

        Only meaningful because `surrogate_scale` is separate from
        `temperatures`: with the two coupled this would flatten the candidate
        softmax of every node holding an `eq`, which is what made the emulated
        fix need a compensating learning rate. Returns the nodes it affects.
        """
        self.carrier_scaled=True
        return tuple(n.name for n in self.program.nodes
                     if any(c.operator.name=="eq" and c.operator.gradient!="none" and
                            carrier_temperature(c.operator.inputs[0])>1. for c in n.candidates))
    def selections(self): return {n.name:self.frozen.get(n.name,int(p.argmax())) for n,p in zip(self.program.nodes,self.choices)}
    def export(self):
        constants=tuple((k,Value.unflat(v.type,self.constants[k].detach().cpu().tolist()) if k in self.constants else v) for k,v in self.program.constants)
        return replace(self.program.harden(self.selections()),constants=constants,trainable_constants=()).validate(self.registry)
    def freeze(self,name,index=None):
        i=next(i for i,n in enumerate(self.program.nodes) if n.name==name)
        self.frozen[name]=int(self.choices[i].argmax()) if index is None else index
        self.choices[i].requires_grad_(False); self.trials.pop(name,None)
        for key,param in self.constants.items():
            users=[n.name for n in self.program.nodes if any(key in c.sources for c in n.candidates)]
            if users and all(u in self.frozen for u in users): param.requires_grad_(False)
    def probe_loss(self,trace,targets,signals):
        self.program.validate_signals(signals); total=torch.tensor(0.,device=next(iter(trace.values())).device)
        for s in signals:
            pred=trace[s.source]; target=targets[s.target]
            if isinstance(target,Value): target=tensor(target,pred.device)
            if s.loss=="mse": loss=(pred-target).square().mean()
            elif s.loss=="bce": loss=nn.functional.binary_cross_entropy(pred.clamp(1e-6,1-1e-6),target)
            else: loss=nn.functional.cross_entropy(pred,target.long().squeeze(-1))
            total=total+s.weight*loss
        return total
