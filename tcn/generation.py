"""One typed, visibility-enforcing, replayable host for all synthetic generators."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import copy
import gzip
import hashlib
import importlib
import json
import math
import random
import os
from functools import lru_cache
from .types import Type, Value, BOOL, integer, floating, product

BYTE=integer(8,signed=False,role="byte")
SCALAR=floating()

def text_value(text,capacity=256):
    data=text.encode("utf8")
    if len(data)>capacity: raise OverflowError("text capacity exceeded")
    return Value.of(product(integer(32,signed=False,bounds=(0,capacity)),product(*(BYTE for _ in range(capacity)))),(len(data),tuple(data)+tuple(0 for _ in range(capacity-len(data)))))
def read_text(v):
    length,data=v.decoded
    return bytes(data[:length]).decode("utf8",errors="replace")
def vector_value(xs): return Value.of(product(*(SCALAR for _ in xs)),tuple(float(x) for x in xs))
def image_value(image):
    import numpy as np
    a=np.asarray(image,dtype=np.uint8)
    return Value.of(product(integer(16,signed=False),integer(16,signed=False),integer(8,signed=False),product(*(BYTE for _ in range(a.size)))),(a.shape[0],a.shape[1],a.shape[2] if a.ndim==3 else 1,tuple(int(x) for x in a.flat)))

@lru_cache(maxsize=1)
def source_fingerprint():
    root=Path(__file__).resolve().parent.parent;digest=hashlib.sha256()
    for base in ('tcn','generators'):
        for folder,dirs,files in os.walk(root/base):
            dirs[:]=sorted(d for d in dirs if d not in {'node_modules','__pycache__','dist','.state'})
            for name in sorted(files):
                p=Path(folder)/name
                if p.suffix not in {'.py','.ts','.json'} or '.test.' in name:continue
                digest.update(str(p.relative_to(root)).encode());digest.update(p.read_bytes())
    return digest.hexdigest()

def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts,sort_keys=True,separators=(",",":")).encode()).digest()[:8],"little")

@dataclass(frozen=True)
class Address:
    generator: str
    seed: int = 0
    index: int = 0
    split: str = "train"
    def rng(self,stream="initial",tick=0): return random.Random(stable_seed(asdict(self),stream,tick))

@dataclass(frozen=True)
class Action:
    verb: str
    target: str = ""
    arguments: tuple[tuple[str,Value], ...] = ()
    actor: str = "agent_0"
    def arg(self,name,default=None): return dict(self.arguments)[name].decoded if name in dict(self.arguments) else default
    def to_dict(self): return {"verb":self.verb,"target":self.target,"actor":self.actor,"arguments":{k:v.to_dict() for k,v in self.arguments}}
    @classmethod
    def from_dict(cls,d): return cls(d["verb"],d.get("target",""),tuple((k,Value.from_dict(v)) for k,v in d.get("arguments",{}).items()),d.get("actor","agent_0"))

@dataclass(frozen=True)
class TimedValue:
    value: Value
    time: float
    available_at: float

@dataclass(frozen=True)
class ActorView:
    observations: dict[str,Value]
    available_actions: tuple[str,...]
    time: float
    objective: dict

@dataclass
class StepRecord:
    observations: dict[str,TimedValue]
    actions: tuple[Action,...]
    latent_states: dict[str,Value]
    probes: dict[str,Value]
    available_actions: dict[str,tuple[str,...]]
    transition: dict[str,Value]
    reward_components: dict[str,Value]
    metadata: dict
    time: float
    done: bool = False
    def actor_view(self,actor="agent_0",objective=None):
        visible={}
        for key,t in self.observations.items():
            owner,sep,name=key.partition("/")
            if sep and owner not in {actor,"public"}: continue
            if t.available_at<=self.time:
                visible[name if sep else key]=t.value
        return ActorView(visible,tuple(self.available_actions.get(actor,())),self.time,copy.deepcopy(objective or {}))
    def to_dict(self):
        return {"observations":{k:{"value":v.value.to_dict(),"time":v.time,"available_at":v.available_at} for k,v in self.observations.items()},"actions":[a.to_dict() for a in self.actions],**{name:{k:v.to_dict() for k,v in getattr(self,name).items()} for name in ("latent_states","probes","transition","reward_components")},"available_actions":self.available_actions,"metadata":self.metadata,"time":self.time,"done":self.done}
    @classmethod
    def from_dict(cls,d):
        return cls({k:TimedValue(Value.from_dict(v["value"]),v["time"],v["available_at"]) for k,v in d["observations"].items()},tuple(Action.from_dict(a) for a in d["actions"]),**{name:{k:Value.from_dict(v) for k,v in d[name].items()} for name in ("latent_states","probes","transition","reward_components")},available_actions={k:tuple(v) for k,v in d["available_actions"].items()},metadata=d["metadata"],time=d["time"],done=d["done"])

class Generator:
    """Mechanism subclasses own only state evolution and readouts."""
    version="1"
    action_schema={"wait":{}}
    def configure(self,configuration): pass
    def initialize(self,address,configuration): raise NotImplementedError
    def advance(self,state,actions,dt,rng): raise NotImplementedError
    def observe(self,state): raise NotImplementedError
    def record(self,state,actions=(),transition=None,rewards=None):
        observations,latents,probes,menus=self.observe(state)
        time=state["time"]
        return StepRecord({k:(v if isinstance(v,TimedValue) else TimedValue(v,time,time)) for k,v in observations.items()},tuple(actions),latents,probes,menus,transition or {},{k:Value.of(SCALAR,float(v)) for k,v in (rewards or {}).items()},{"tick":state["tick"]},time,bool(state.get("done",False)))
    def validate_actions(self,actions):
        used=set()
        for a in actions:
            if a.verb not in self.action_schema: raise ValueError(f"unknown action {a.verb}")
            spec=self.action_schema[a.verb]
            args=dict(a.arguments)
            if set(args)!=set(spec): raise TypeError("action argument names mismatch")
            for k,t in spec.items():
                if args[k].type!=t: raise TypeError(f"action argument {k} type mismatch")
            effector=(a.actor,a.verb if a.verb=="say" else "body")
            if effector in used: raise ValueError("conflicting simultaneous action commands")
            used.add(effector)

class Host:
    def __init__(self,generator,address,configuration=None,objective=None):
        self.generator=generator; self.address=address; self.configuration=copy.deepcopy(configuration or {}); self.objective=copy.deepcopy(objective or {})
        generator.configure(self.configuration)
        self.state=generator.initialize(address,self.configuration | {"objective":self.objective})
        self.records=[generator.record(self.state)]; self.inputs=[]
    @classmethod
    def create(cls,name,seed=0,index=0,split="train",configuration=None,objective=None):
        g=load_generator(name)
        return cls(g,Address(name,seed,index,split),configuration,objective)
    def view(self,actor="agent_0"): return self.records[-1].actor_view(actor,self.objective)
    def step(self,actions=(),dt=1.,random_input=0):
        if not dt>0 or not math.isfinite(dt): raise ValueError("dt must be finite and positive")
        if self.records[-1].done: raise ValueError("episode already terminated")
        actions=tuple(actions); self.generator.validate_actions(actions)
        before=copy.deepcopy(self.state)
        rng=self.address.rng(f"transition:{random_input}",before["tick"])
        state,transition,rewards=self.generator.advance(copy.deepcopy(before),actions,dt,rng)
        state["time"]=before["time"]+dt; state["tick"]=before["tick"]+1
        record=self.generator.record(state,actions,transition,rewards)
        self.state=state; self.records.append(record)
        self.inputs.append({"actions":[a.to_dict() for a in actions],"dt":dt,"random_input":random_input})
        return record
    def snapshot(self):
        return copy.deepcopy({"format":"tcn.episode/1","source":source_fingerprint(),"address":asdict(self.address),"version":self.generator.version,"configuration":self.configuration,"objective":self.objective,"state":self.state,"inputs":self.inputs,"records":[r.to_dict() for r in self.records]})
    @classmethod
    def restore(cls,d):
        if d.get("format")!="tcn.episode/1": raise ValueError("unknown episode format")
        if d.get("source")!=source_fingerprint():raise ValueError("episode source revision mismatch")
        address=Address(**d["address"]); gen=load_generator(address.generator)
        if gen.version!=d["version"]: raise ValueError("generator version mismatch")
        obj=cls.__new__(cls); obj.generator=gen; obj.address=address; obj.configuration=copy.deepcopy(d["configuration"]); obj.objective=copy.deepcopy(d["objective"])
        gen.configure(obj.configuration)
        obj.state=copy.deepcopy(d["state"]); obj.inputs=copy.deepcopy(d["inputs"]); obj.records=[StepRecord.from_dict(r) for r in d["records"]]
        return obj
    def replay(self):
        h=Host(self.generator,self.address,self.configuration,self.objective)
        for row in self.inputs: h.step(tuple(Action.from_dict(a) for a in row["actions"]),row["dt"],row["random_input"])
        if h.digest!=self.digest: raise AssertionError("replayed episode diverged")
        return h
    @property
    def digest(self): return hashlib.sha256(json.dumps(self.snapshot(),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    def save(self,path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        opener=gzip.open if path.suffix==".gz" else open
        with opener(path,"wt") as f: json.dump(self.snapshot(),f,sort_keys=True,allow_nan=False)
    @classmethod
    def load(cls,path):
        opener=gzip.open if str(path).endswith(".gz") else open
        with opener(path,"rt") as f: return cls.restore(json.load(f))

def generator_root(): return Path(__file__).resolve().parent.parent/"generators"
def manifests():
    return {p.parent.name:json.loads(p.read_text()) for p in sorted(generator_root().glob("*/manifest.json"))}
def load_generator(name):
    specs=manifests()
    if name not in specs: raise KeyError(f"unknown generator {name}")
    module,entry=specs[name]["entrypoint"].split(":")
    instance=getattr(importlib.import_module(module),entry)()
    if not isinstance(instance,Generator): raise TypeError("entrypoint must implement Generator")
    return instance
