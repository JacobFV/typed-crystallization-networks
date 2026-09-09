"""Immutable typed DAGs, explicit recurrence, and content-addressed module versions."""
from __future__ import annotations
from dataclasses import dataclass, replace
import hashlib
import itertools
import json
from .types import Type, Value
from .operators import Registry, Operator

@dataclass(frozen=True)
class Candidate:
    operator: Operator
    sources: tuple[str, ...]
    def to_dict(self): return {"operator":self.operator.to_dict(),"sources":list(self.sources)}

@dataclass(frozen=True)
class Node:
    name: str
    output: Type
    candidates: tuple[Candidate, ...]
    region: str = "core"
    depth: int = 1
    selected: int | None = None
    def __post_init__(self):
        if not self.candidates: raise ValueError("node has no legal candidates")
        if self.selected is not None and not 0<=self.selected<len(self.candidates): raise ValueError("invalid selection")

@dataclass(frozen=True)
class Signal:
    source: str
    target: str
    regions: tuple[str, ...]
    type: Type
    loss: str = "mse"
    horizon: int = 0
    weight: float = 1.0
    def __post_init__(self):
        if self.loss not in {"mse","bce","categorical"}: raise ValueError("unsupported loss")
        if self.horizon < 0 or self.weight < 0: raise ValueError("invalid supervision configuration")

@dataclass(frozen=True)
class Program:
    inputs: tuple[tuple[str, Type], ...]
    nodes: tuple[Node, ...]
    outputs: tuple[tuple[str, str], ...]
    constants: tuple[tuple[str, Value], ...] = ()
    state: tuple[tuple[str, Value, str], ...] = ()
    input_depths: tuple[tuple[str, int], ...] = ()
    version: int = 1
    trainable_constants: tuple[str, ...] = ()
    def port_types(self):
        return dict(self.inputs) | {k:v.type for k,v in self.constants} | {k:v.type for k,v,_ in self.state} | {n.name:n.output for n in self.nodes}
    @property
    def is_frozen(self): return all(n.selected is not None for n in self.nodes) and not self.trainable_constants
    def validate(self, registry=None):
        registry=registry or Registry(); known={}; depths=dict(self.input_depths)
        if len({k for k,_ in self.outputs}) != len(self.outputs): raise ValueError("duplicate output names")
        for name,t in self.inputs:
            if name in known: raise ValueError("duplicate port")
            known[name]=t; depths.setdefault(name,0)
        for name,v in self.constants:
            if name in known: raise ValueError("duplicate constant")
            known[name]=v.type; depths[name]=-1
        for name,v,_ in self.state:
            if name in known: raise ValueError("duplicate state")
            known[name]=v.type; depths[name]=-1
        if not set(self.trainable_constants)<=set(dict(self.constants)): raise ValueError("unknown trainable constant")
        for name in self.trainable_constants:
            if not dict(self.constants)[name].type.numeric: raise TypeError("trainable constant requires numeric encoding")
        for n in self.nodes:
            if n.name in known: raise ValueError("duplicate node")
            for c in n.candidates:
                if any(s not in known for s in c.sources): raise ValueError("forward reference or instantaneous cycle")
                if any(depths[s]>=n.depth for s in c.sources): raise ValueError("edge violates depth scaffold")
                if tuple(known[s] for s in c.sources)!=c.operator.inputs: raise TypeError("candidate input mismatch")
                if c.operator.output!=n.output: raise TypeError("mixed output types")
                if registry.from_dict(c.operator.to_dict())!=c.operator: raise TypeError("operator contract drift")
            known[n.name]=n.output; depths[n.name]=n.depth
        for _,source in self.outputs:
            if source not in known: raise ValueError("unknown output source")
        for name,v,update in self.state:
            if update not in known or known[update]!=v.type: raise TypeError("state update mismatch")
        return self
    def validate_signals(self, signals):
        types=self.port_types(); regions={n.name:n.region for n in self.nodes}
        for s in signals:
            if types.get(s.source)!=s.type: raise TypeError("probe signature mismatch")
            if s.source in regions and regions[s.source] not in s.regions: raise ValueError("probe outside admissible region")
    def execute(self, inputs, state=None, registry=None, selections=None):
        r=registry or Registry()
        # A Program is immutable, so validity against a given registry cannot change
        # once established. Module operators execute a whole sub-program per batch
        # row, and re-validating that body every row dominated their cost; cache the
        # result on the instance instead. Registering further modules cannot
        # invalidate an already-checked program, since its own candidates are fixed.
        if getattr(self,'_validated',None) is not r:
            self.validate(r); object.__setattr__(self,'_validated',r)
        if set(inputs)!=set(dict(self.inputs)): raise ValueError("input port set mismatch")
        if any(inputs[k].type!=t for k,t in self.inputs): raise TypeError("input representation mismatch")
        values=dict(inputs)|dict(self.constants)
        if state is not None and set(state)!= {k for k,_,_ in self.state}: raise ValueError("state port set mismatch")
        for k,v,_ in self.state:
            x=v if state is None else state[k]
            if x.type!=v.type: raise TypeError("state type mismatch")
            values[k]=x
        for n in self.nodes:
            i=n.selected if n.selected is not None else (selections or {}).get(n.name)
            if i is None: raise ValueError("exact execution needs discrete choices")
            c=n.candidates[i]
            values[n.name]=r.exact(c.operator,[values[k] for k in c.sources])
        return {k:values[v] for k,v in self.outputs},{k:values[u] for k,_,u in self.state},values
    def run(self, inputs, state=None, registry=None, selections=None):
        out,st,_=self.execute(inputs,state,registry,selections); return out,st
    def lift_state(self):
        """Expose recurrence as ordinary module inputs and outputs for reuse."""
        return replace(self,inputs=self.inputs+tuple((k,v.type) for k,v,_ in self.state),outputs=self.outputs+tuple(('state:'+k,u) for k,_,u in self.state),state=(),input_depths=self.input_depths+tuple((k,-1) for k,_,_ in self.state),version=self.version+1)
    def harden(self, selections):
        nodes=[]
        for n in self.nodes:
            i=n.selected if n.selected is not None else selections.get(n.name)
            nodes.append(n if i is None else replace(n,candidates=(n.candidates[i],),selected=0))
        return replace(self,nodes=tuple(nodes),version=self.version+1)
    def pruned(self):
        """Drop nodes that cannot influence any output or state update.

        A hardened export keeps the whole scaffold, so its description bits and
        execution cost describe the scaffold rather than the discovered program,
        and a registered module's dead gates are charged transitively at every
        call site for the life of the module. Reachability is taken over the
        selected candidate where a node is hardened and over *every* candidate
        where it is not, so removal is semantics-preserving for a soft scaffold
        as well as for a frozen program. Constants and state ports are part of
        the declared interface and are left alone. This changes the digest.
        """
        by={n.name:n for n in self.nodes}; keep=set()
        stack=[v for _,v in self.outputs]+[u for _,_,u in self.state]
        while stack:
            k=stack.pop()
            if k in by and k not in keep:
                keep.add(k); n=by[k]
                live=(n.candidates[n.selected],) if n.selected is not None else n.candidates
                stack+=[s for c in live for s in c.sources]
        return replace(self,nodes=tuple(n for n in self.nodes if n.name in keep))
    def to_dict(self):
        return {"format":"tcn.program/1","version":self.version,"trainable_constants":list(self.trainable_constants),"inputs":[[k,t.to_dict()] for k,t in self.inputs],"nodes":[{"name":n.name,"output":n.output.to_dict(),"candidates":[c.to_dict() for c in n.candidates],"region":n.region,"depth":n.depth,"selected":n.selected} for n in self.nodes],"outputs":list(self.outputs),"constants":[[k,v.to_dict()] for k,v in self.constants],"state":[[k,v.to_dict(),u] for k,v,u in self.state],"input_depths":list(self.input_depths)}
    @classmethod
    def from_dict(cls,d,registry=None):
        if d.get("format")!="tcn.program/1": raise ValueError("unsupported program format")
        r=registry or Registry()
        nodes=tuple(Node(n["name"],Type.from_dict(n["output"]),tuple(Candidate(r.from_dict(c["operator"]),tuple(c["sources"])) for c in n["candidates"]),n["region"],n["depth"],n["selected"]) for n in d["nodes"])
        return cls(tuple((k,Type.from_dict(t)) for k,t in d["inputs"]),nodes,tuple(map(tuple,d["outputs"])),tuple((k,Value.from_dict(v)) for k,v in d["constants"]),tuple((k,Value.from_dict(v),u) for k,v,u in d["state"]),tuple(map(tuple,d.get("input_depths",()))),d["version"],tuple(d.get("trainable_constants",()))).validate(r)
    @property
    def digest(self): return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True,separators=(",",":")).encode()).hexdigest()[:24]
    def execution_cost(self,registry=None):
        return sum(n.candidates[n.selected].operator.cost if n.selected is not None else max(c.operator.cost for c in n.candidates) for n in self.nodes)
    def description_bits(self,registry=None):
        size=len(json.dumps(self.to_dict(),sort_keys=True).encode())*8
        if registry:
            seen=set()
            def definitions(program):
                total=0
                for node in program.nodes:
                    for c in node.candidates:
                        names=[c.operator.name,dict(c.operator.parameters).get('module','')]
                        for name in names:
                            if name.startswith('module:') and name not in seen:
                                seen.add(name);module=registry.modules[name]
                                total+=module.description_bits()+definitions(module)
                return total
            size+=definitions(self)
        return size

def legal_candidates(registry,names,ports,output,arities=(1,2),limit=4096):
    """Enumerate bounded legal wiring/operation choices; no learned coercions."""
    out=[]; attempted=0
    for name in names:
        for arity in arities:
            for refs in itertools.product(ports,repeat=arity):
                attempted+=1
                if attempted>limit: raise ValueError("candidate enumeration budget exceeded")
                try: op=registry.resolve(name,[ports[k] for k in refs],output)
                except (TypeError,KeyError,ValueError): continue
                out.append(Candidate(op,refs))
    return tuple(out)
