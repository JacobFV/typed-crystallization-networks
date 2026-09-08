"""Four carriers, explicit representations, canonical immutable values."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import itertools
import json
import math
import struct

@dataclass(frozen=True)
class Encoding:
    kind: str = "integer"
    signed: bool = True
    scale: float = 1.0
    overflow: str = "error"
    def __post_init__(self):
        if self.kind not in {"integer", "fixed", "float"}:
            raise ValueError("unknown encoding")
        if self.scale <= 0 or not math.isfinite(self.scale):
            raise ValueError("scale must be positive and finite")
        if self.overflow not in {"error", "wrap", "saturate"}:
            raise ValueError("unknown overflow contract")

@dataclass(frozen=True)
class Type:
    kind: str
    bits: int = 0
    items: tuple[Type, ...] = ()
    encoding: Encoding = Encoding()
    role: str = ""
    unit: str = ""
    frame: str = ""
    bounds: tuple[float, float] | None = None
    capacity: int = 0
    def __post_init__(self):
        if self.kind not in {"bool", "int", "tuple", "set"}:
            raise ValueError("only bool, int[n], tuple and set are carriers")
        if self.kind == "int" and not 1 <= self.bits <= 64:
            raise ValueError("integer width must be 1..64")
        if self.kind == "int" and self.encoding.kind == "float" and self.bits not in (32, 64):
            raise ValueError("floating encoding requires 32 or 64 bits")
        if self.kind == "set" and (len(self.items) != 1 or self.capacity < 0):
            raise ValueError("set requires one element type and finite capacity")
        if self.kind in {"bool", "int"} and self.items:
            raise ValueError("scalar cannot contain fields")
        if self.bounds is not None and (len(self.bounds) != 2 or self.bounds[0] > self.bounds[1]):
            raise ValueError("invalid refinement bounds")
    @property
    def numeric(self):
        return self.kind == "int" and self.role not in {"category", "symbol", "byte"}
    def to_dict(self):
        d={"kind":self.kind}
        if self.kind=="int":d.update(bits=self.bits,encoding=asdict(self.encoding))
        if self.items:d["items"]=[t.to_dict() for t in self.items]
        if self.kind=="set":d["capacity"]=self.capacity
        for key in ("role","unit","frame","bounds"):
            if getattr(self,key):d[key]=getattr(self,key)
        return d
    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["encoding"] = Encoding(**d.get("encoding", {}))
        d["items"] = tuple(cls.from_dict(t) for t in d.get("items", ()))
        if d.get("bounds") is not None:
            d["bounds"] = tuple(d["bounds"])
        return cls(**d)
    def universe(self, limit=256):
        if self.kind == "bool":
            return (False, True)
        if self.kind == "int" and self.encoding.kind == "integer" and 2**self.bits <= limit:
            return tuple(range(2**self.bits))
        if self.kind == "tuple":
            us = [t.universe(limit) for t in self.items]
            if math.prod(map(len, us)) <= limit:
                return tuple(itertools.product(*us))
        raise ValueError("finite relaxation universe exceeds configured limit")
    @property
    def width(self):
        if self.kind in {"bool", "int"}:
            return self.bits if self.kind=="int" and self.role in {"category","symbol"} else 1
        if self.kind == "tuple":
            return sum(t.width for t in self.items)
        return self.capacity*(1+self.items[0].width)

BOOL = Type("bool")
def integer(bits=16, *, signed=True, overflow="error", **schema):
    return Type("int", bits, encoding=Encoding(signed=signed, overflow=overflow), **schema)
def floating(bits=32, **schema):
    return Type("int", bits, encoding=Encoding("float"), **schema)
def fixed(bits=16, scale=256, *, overflow="error", **schema):
    return Type("int", bits, encoding=Encoding("fixed", True, scale, overflow), **schema)
def product(*types, **schema):
    return Type("tuple", items=tuple(types), **schema)
def setof(element, capacity=64, **schema):
    return Type("set", items=(element,), capacity=capacity, **schema)

def canonical(x):
    if isinstance(x, frozenset):
        return sorted((canonical(v) for v in x), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(x, tuple):
        return [canonical(v) for v in x]
    return x

def encode(t, x):
    if t.kind == "bool":
        if type(x) is not bool:
            raise TypeError("Boolean encoding requires bool, not numeric coercion")
        return x
    if t.kind == "tuple":
        if len(x) != len(t.items):
            raise ValueError("tuple arity mismatch")
        return tuple(encode(a, b) for a, b in zip(t.items, x))
    if t.kind == "set":
        out = frozenset(encode(t.items[0], v) for v in x)
        if len(out) > t.capacity:
            raise OverflowError("set capacity exceeded")
        return out
    if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x):
        raise TypeError("finite numeric value required")
    if t.bounds and not t.bounds[0] <= x <= t.bounds[1]:
        raise ValueError("value outside semantic bounds")
    e = t.encoding
    if e.kind == "float":
        fmt = ("f", "I") if t.bits == 32 else ("d", "Q")
        raw = struct.unpack("<" + fmt[1], struct.pack("<" + fmt[0], x))[0]
        if not math.isfinite(decode(t, raw)):
            raise OverflowError("nonfinite float encoding")
        return raw
    n = round(x * e.scale) if e.kind == "fixed" else int(x)
    if e.kind == "integer" and n != x:
        raise ValueError("fractional value requires explicit quantization")
    lo = -(2**(t.bits-1)) if e.signed else 0
    hi = 2**(t.bits-int(e.signed))-1
    if not lo <= n <= hi:
        if e.overflow == "error":
            raise OverflowError(f"{n} outside [{lo}, {hi}]")
        if e.overflow == "saturate":
            n = min(hi, max(lo, n))
    return n % 2**t.bits

def decode(t, raw):
    if t.kind == "bool":
        return raw
    if t.kind == "tuple":
        return tuple(decode(a, b) for a, b in zip(t.items, raw))
    if t.kind == "set":
        return frozenset(decode(t.items[0], v) for v in raw)
    e = t.encoding
    if e.kind == "float":
        fmt = ("I", "f") if t.bits == 32 else ("Q", "d")
        return struct.unpack("<" + fmt[1], struct.pack("<" + fmt[0], raw))[0]
    n = raw - 2**t.bits if e.signed and raw >= 2**(t.bits-1) else raw
    return n / e.scale if e.kind == "fixed" else n

def validate_raw(t, raw):
    if t.kind == "bool":
        if type(raw) is not bool: raise TypeError("invalid Boolean carrier")
    elif t.kind == "int":
        if type(raw) is not int or not 0 <= raw < 2**t.bits: raise TypeError("invalid integer carrier")
        x = decode(t, raw)
        if not math.isfinite(x) or (t.bounds and not t.bounds[0] <= x <= t.bounds[1]):
            raise ValueError("encoded value violates domain")
    elif t.kind == "tuple":
        if type(raw) is not tuple or len(raw) != len(t.items): raise TypeError("invalid tuple")
        for a, b in zip(t.items, raw): validate_raw(a, b)
    else:
        if type(raw) is not frozenset or len(raw) > t.capacity: raise TypeError("invalid set")
        for x in raw: validate_raw(t.items[0], x)

@dataclass(frozen=True)
class Value:
    type: Type
    raw: object
    def __post_init__(self): validate_raw(self.type, self.raw)
    @classmethod
    def of(cls, t, x): return cls(t, encode(t, x))
    @property
    def decoded(self): return decode(self.type, self.raw)
    def to_dict(self): return {"type": self.type.to_dict(), "raw": canonical(self.raw)}
    @classmethod
    def from_dict(cls, d):
        t = Type.from_dict(d["type"])
        def restore(t, x):
            if t.kind == "tuple": return tuple(restore(a, b) for a, b in zip(t.items, x))
            if t.kind == "set": return frozenset(restore(t.items[0], v) for v in x)
            return x
        return cls(t, restore(t, d["raw"]))
    def flat(self):
        def f(t, raw):
            if t.kind in {"bool", "int"}:
                if t.kind=="int" and t.role in {"category","symbol"}:return [float((raw>>i)&1) for i in range(t.bits)]
                return [float(decode(t, raw))]
            if t.kind == "tuple": return [v for a,b in zip(t.items,raw) for v in f(a,b)]
            ordered=sorted(raw,key=lambda v:json.dumps(canonical(v),sort_keys=True))
            present=[z for x in ordered for z in [1.,*f(t.items[0],x)]]
            return present+[0.] * ((t.capacity-len(raw))*(1+t.items[0].width))
        return f(self.type, self.raw)
    @classmethod
    def unflat(cls, t, flat):
        it = iter(flat)
        def f(t):
            if t.kind == "bool": return bool(next(it) >= .5)
            if t.kind == "int":
                if t.role in {"category","symbol"}:
                    raw=sum(int(next(it)>=.5)<<i for i in range(t.bits))
                    return decode(t,raw)
                x = float(next(it))
                return round(x) if t.encoding.kind == "integer" else x
            if t.kind == "tuple": return tuple(f(a) for a in t.items)
            vals=[]
            for _ in range(t.capacity):
                if next(it)>=.5:vals.append(f(t.items[0]))
                else:
                    for _ in range(t.items[0].width):next(it)
            return frozenset(vals)
        v = cls.of(t, f(t))
        if next(it, None) is not None: raise ValueError("unused flattened fields")
        return v
