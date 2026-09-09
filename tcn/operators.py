"""Universal typed operator registry; exact execution uses only the standard library."""
from __future__ import annotations
from dataclasses import dataclass, replace
import cmath
import math
from .types import Type, Value, BOOL, integer, product, setof, decode, interpretable

BINARY = {"add", "sub", "mul", "div", "pow", "mod", "idiv", "min", "max", "shl", "shr", "atan2"}
UNARY = {"neg", "abs", "exp", "log", "sin", "cos", "sqrt"}
COMPARE = {"eq", "lt", "le", "gt", "ge"}
LOGIC = {"and", "or", "xor", "nand", "nor", "xnor"}
STRUCTURAL = {"tuple", "project", "index", "member", "insert", "remove", "union", "intersection", "pair", "join", "map", "filter"}
CONVERSIONS = {"encode", "decode", "quantize", "dequantize", "pack", "unpack", "interpret"}

@dataclass(frozen=True)
class Operator:
    name: str
    inputs: tuple[Type, ...]
    output: Type
    parameters: tuple = ()
    gradient: str = "exact"
    cost: float = 1.0
    def to_dict(self):
        return {"name": self.name, "inputs": [t.to_dict() for t in self.inputs], "output": self.output.to_dict(), "parameters": dict(self.parameters)}

class Registry:
    def __init__(self): self.modules = {}
    def register_module(self, module):
        # Register the program the module actually computes, not the scaffold it
        # was learned in: a dead gate kept by `SoftProgram.export()` would be
        # charged to description size and execution cost at every call site for
        # the life of the module. Pruning preserves exact semantics.
        module = module.pruned()
        name = "module:" + module.digest
        if not module.is_frozen: raise ValueError("only crystallized modules are callable operators")
        if module.state:
            module=module.lift_state();name="module:"+module.digest
        self.modules[name] = module
        return name
    def resolve(self, name, inputs, output=None, parameters=None):
        ts = tuple(inputs); p = dict(parameters or {}); grad = "exact"; cost = 1.0
        def require(ok, message="operator signature mismatch"):
            if not ok: raise TypeError(f"{name}: {message}")
        if name.startswith("module:"):
            require(name in self.modules, "unknown frozen module")
            m = self.modules[name]
            require(ts == tuple(t for _,t in m.inputs))
            # A single-output module yields that output's type directly. Wrapping it
            # in a one-field product forced a `project` node at every call site,
            # which made calling a module strictly costlier than inlining its body
            # for any number of call sites. Multi-output modules still form a tuple.
            outs = tuple(m.port_types()[v] for _,v in m.outputs)
            inferred = outs[0] if len(outs) == 1 else product(*outs)
            grad = "none"; cost = m.execution_cost(self)
        elif name == "identity":
            require(len(ts) == 1); inferred = ts[0]
        elif name in LOGIC:
            require(ts == (BOOL, BOOL)); inferred = BOOL
        elif name == "not":
            require(ts == (BOOL,)); inferred = BOOL
        elif name.startswith("truth_"):
            require(ts == (BOOL, BOOL) and 0 <= int(name[6:]) < 16); inferred = BOOL
        elif name == "mux":
            require(len(ts)==3 and ts[0]==BOOL and ts[1]==ts[2]); inferred = ts[1]
        elif name in COMPARE:
            require(len(ts)==2 and ts[0]==ts[1])
            require(name=="eq" or ts[0].numeric); inferred=BOOL; grad="surrogate"
        elif name in BINARY | UNARY:
            require(len(ts)==(2 if name in BINARY else 1) and all(t.numeric for t in ts))
            require(len(ts)==1 or ts[0]==ts[1])
            t=ts[0]; inferred=t
            if name in {"exp","log","sin","cos","pow","atan2"}:
                require(not t.unit, "analytic argument must be dimensionless")
            if name in {"mod","idiv","shl","shr"}:
                require(t.encoding.kind=="integer"); grad="surrogate"
            if name=="mul" and t.unit: inferred=replace(t, unit=f"({t.unit})^2", bounds=None)
            if name=="div": inferred=replace(t, unit="", bounds=None)
            if name=="sqrt" and t.unit: inferred=replace(t, unit=f"sqrt({t.unit})", bounds=None)
            if name in {"div","pow","exp","log","sin","cos","atan2","sqrt"}:
                require(t.encoding.kind != "integer", "analytic operation requires an explicit numeric encoding")
        elif name in {"sum","mean","reduce_min","reduce_max","count"}:
            require(len(ts)==1 and ts[0].kind in {"tuple","set"})
            elems=ts[0].items
            if name=="count":
                inferred=integer(32, signed=False)
                if ts[0].kind=="set":grad="none"
            else:
                require(bool(elems) and all(t==elems[0] and t.numeric for t in elems))
                inferred=elems[0]
                if name=="mean": require(inferred.encoding.kind!="integer")
                if ts[0].kind=="set": grad="none"
        elif name=="tuple": inferred=product(*ts)
        elif name=="project":
            require(len(ts)==1 and ts[0].kind=="tuple" and 0<=p.get("index",-1)<len(ts[0].items))
            inferred=ts[0].items[p["index"]]
        elif name=="index":
            require(len(ts)==2 and ts[0].kind=="tuple" and ts[1].kind=="int" and ts[1].encoding.kind=="integer")
            require(bool(ts[0].items) and len(set(ts[0].items))==1)
            inferred=ts[0].items[0]; grad="surrogate"
        elif name in {"member","insert","remove"}:
            require(len(ts)==2 and ts[0].kind=="set" and ts[0].items[0]==ts[1])
            inferred=BOOL if name=="member" else ts[0]; grad="none"
        elif name in {"union","intersection"}:
            require(len(ts)==2 and ts[0]==ts[1] and ts[0].kind=="set"); inferred=ts[0]; grad="none"
        elif name in {"pair","join"}:
            require(len(ts)==2 and all(t.kind=="set" for t in ts))
            inferred=setof(product(ts[0].items[0],ts[1].items[0]), ts[0].capacity*ts[1].capacity)
            if name=="join": require("left" in p and "right" in p)
            grad="none"
        elif name in {"map","filter"}:
            require(len(ts)==1 and ts[0].kind=="set" and p.get("module") in self.modules)
            m=self.modules[p["module"]]
            require(tuple(t for _,t in m.inputs)==ts[0].items)
            outs=tuple(m.port_types()[v] for _,v in m.outputs)
            require(len(outs)==1)
            if name=="filter": require(outs==(BOOL,))
            inferred=ts[0] if name=="filter" else setof(outs[0],ts[0].capacity)
            grad="none"; cost=max(1,ts[0].capacity)*m.execution_cost(self)
        elif name in {"delay","difference","accumulation"}:
            require(len(ts)==2 and ts[0]==ts[1])
            if name!="delay": require(ts[0].numeric)
            inferred=product(ts[0],ts[0])
        elif name in {"fft","ifft"}:
            require(len(ts)==1 and ts[0].kind=="tuple" and len(ts[0].items)>0)
            t=ts[0].items[0]
            require(all(x==t for x in ts[0].items))
            if name=="fft":
                require(t.numeric and t.encoding.kind!="integer")
                inferred=product(*(product(t,t) for _ in ts[0].items))
            else:
                require(t.kind=="tuple" and len(t.items)==2 and t.items[0]==t.items[1] and t.items[0].numeric)
                inferred=ts[0]
            cost=len(ts[0].items)*max(1,math.log2(len(ts[0].items)))
        elif name=="fourier_basis":
            require(len(ts)==1 and ts[0].numeric and not ts[0].unit)
            inferred=product(ts[0],ts[0])
        elif name in CONVERSIONS:
            require(len(ts)==1 and output is not None, "representation output must be explicit")
            inferred=output
            if name=="interpret":
                # A declared semantic commitment, not a representation change: the
                # carrier, encoding, unit, frame and bounds are all preserved and
                # only `role` moves, from uncommitted to committed. Exact and
                # bijective on the raw value, so the error contract is zero error
                # in both directions; there is simply no operator back.
                require(ts[0].kind=="int" and output.kind=="int", "interpretation applies to an integer carrier")
                require(interpretable(ts[0].role, output.role),
                        "interpretation only commits an uncommitted role to a declared one")
                require(replace(ts[0], role=output.role)==output,
                        "interpretation declares a role and preserves the carrier")
                # The gradient follows from the relaxation contract already in
                # `Type.flat`, it is not a separate choice. A magnitude flattens
                # to the same single scalar the uncommitted byte does, so the
                # relaxation is the identity and the derivative is 1. A nominal ID
                # flattens to a bit vector of a different width, and there is no
                # valid derivative from a magnitude into unordered bits: that is
                # an explicit gradient boundary, exactly as ARCHITECTURE section 2
                # requires of a candidate without a valid relaxation.
                grad = "exact" if output.numeric else "none"
            elif name in {"pack","unpack"}:
                tup=ts[0] if name=="pack" else output
                scalar=output if name=="pack" else ts[0]
                require(tup.kind=="tuple" and all(t.kind in {"bool","int"} for t in tup.items))
                require(scalar.kind=="int" and scalar.bits==sum(1 if t.kind=="bool" else t.bits for t in tup.items))
                grad="none"
            else:
                require(ts[0].kind in {"bool","int"} and output.kind in {"bool","int"})
                require(ts[0].unit==output.unit and ts[0].frame==output.frame and ts[0].role==output.role)
                grad="surrogate"
        else: raise KeyError(f"unknown universal operator: {name}")
        require(output is None or output==inferred, "output signature mismatch")
        return Operator(name,ts,inferred,tuple(sorted(p.items())),grad,cost)
    def from_dict(self,d):
        return self.resolve(d["name"], [Type.from_dict(t) for t in d["inputs"]], Type.from_dict(d["output"]), d.get("parameters"))
    def exact(self, op, args):
        if tuple(v.type for v in args)!=op.inputs: raise TypeError("input type mismatch")
        n=op.name; p=dict(op.parameters); xs=[v.decoded for v in args]
        if n.startswith("module:"):
            m=self.modules[n]; out,_=m.run(dict(zip((k for k,_ in m.inputs),args)),registry=self)
            raw=tuple(v.raw for v in out.values())
            return Value(op.output,raw[0] if len(raw)==1 else raw)
        if n=="pack":
            raw=0; offset=0
            for t,x in zip(args[0].type.items,args[0].raw):
                raw |= int(x)<<offset; offset+=1 if t.kind=="bool" else t.bits
            return Value(op.output,raw)
        if n=="unpack":
            out=[]; raw=args[0].raw
            for t in op.output.items:
                b=1 if t.kind=="bool" else t.bits; v=raw & (2**b-1); raw >>= b
                out.append(bool(v) if t.kind=="bool" else v)
            return Value(op.output,tuple(out))
        if n=="identity": return args[0]
        if n=="interpret": return Value(op.output,args[0].raw)
        if n=="not": y=not xs[0]
        elif n in LOGIC:
            a,b=xs; y={"and":a and b,"or":a or b,"xor":a!=b,"nand":not(a and b),"nor":not(a or b),"xnor":a==b}[n]
        elif n.startswith("truth_"): y=bool((int(n[6:])>>(2*int(xs[0])+int(xs[1])))&1)
        elif n=="mux": y=xs[1] if xs[0] else xs[2]
        elif n in COMPARE:
            a,b=xs
            y={"eq":lambda:a==b,"lt":lambda:a<b,"le":lambda:a<=b,"gt":lambda:a>b,"ge":lambda:a>=b}[n]()
        elif n in BINARY:
            a,b=xs
            if n in {"div","mod","idiv"} and b==0: raise ValueError("zero denominator")
            if n in {"shl","shr"} and not 0<=b<op.inputs[0].bits: raise ValueError("shift outside bit width")
            y={"add":lambda:a+b,"sub":lambda:a-b,"mul":lambda:a*b,"div":lambda:a/b,"pow":lambda:math.pow(a,b),"mod":lambda:a%b,"idiv":lambda:a//b,"min":lambda:min(a,b),"max":lambda:max(a,b),"shl":lambda:a<<b,"shr":lambda:a>>b,"atan2":lambda:math.atan2(a,b)}[n]()
        elif n in UNARY:
            y={"neg":lambda x:-x,"abs":abs,"exp":math.exp,"log":math.log,"sin":math.sin,"cos":math.cos,"sqrt":math.sqrt}[n](xs[0])
        elif n in {"sum","mean","reduce_min","reduce_max","count"}:
            a=sorted(xs[0]) if isinstance(xs[0],frozenset) else xs[0]
            if n in {"mean","reduce_min","reduce_max"} and not a: raise ValueError("empty reduction")
            y={"sum":lambda:sum(a),"mean":lambda:sum(a)/len(a),"reduce_min":lambda:min(a),"reduce_max":lambda:max(a),"count":lambda:len(a)}[n]()
        elif n=="tuple": y=tuple(xs)
        elif n in {"project","index"}:
            i=p["index"] if n=="project" else xs[1]
            if not 0<=i<len(xs[0]): raise IndexError("index outside tuple")
            y=xs[0][i]
        elif n=="member": y=xs[1] in xs[0]
        elif n=="insert": y=xs[0] | {xs[1]}
        elif n=="remove": y=xs[0] - {xs[1]}
        elif n=="union": y=xs[0] | xs[1]
        elif n=="intersection": y=xs[0] & xs[1]
        elif n in {"pair","join"}:
            y=frozenset((a,b) for a in xs[0] for b in xs[1] if n=="pair" or a[p["left"]]==b[p["right"]])
        elif n in {"map","filter"}:
            m=self.modules[p["module"]]; vals=[]
            for x in xs[0]:
                result,_=m.run({m.inputs[0][0]:Value.of(op.inputs[0].items[0],x)},registry=self)
                v=next(iter(result.values())).decoded
                if n=="map": vals.append(v)
                elif v: vals.append(x)
            y=frozenset(vals)
        elif n in {"delay","difference","accumulation"}:
            current,old=xs
            y=(old,current) if n=="delay" else ((current-old,current) if n=="difference" else (old+current,old+current))
        elif n in {"fft","ifft"}:
            a=[complex(v) for v in xs[0]] if n=="fft" else [complex(*v) for v in xs[0]]
            sign=-1 if n=="fft" else 1; size=len(a)
            z=[sum(v*cmath.exp(sign*2j*math.pi*k*j/size) for j,v in enumerate(a))/(size if n=="ifft" else 1) for k in range(size)]
            y=tuple((v.real,v.imag) for v in z)
        elif n=="fourier_basis": y=(math.cos(xs[0]),math.sin(xs[0]))
        elif n in CONVERSIONS:
            x=xs[0]
            y=bool(x>=p.get("threshold",.5)) if op.output.kind=="bool" else (round(float(x)) if op.output.encoding.kind=="integer" else float(x))
        else: raise KeyError(n)
        return Value.of(op.output,y)
