from tcn.generation import Generator, Value, BOOL
from tcn.types import integer,product,setof
E=integer(3,signed=False,role='category'); EDGE=product(E,E); EDGES=setof(EDGE,64)

class Implementation(Generator):
    action_schema={'wait':{},'assert':{'pair':EDGE,'value':BOOL}}
    def initialize(self,address,configuration):
        rng=address.rng(); n=int(configuration.get('entities',5))
        if not 1<=n<=8:raise ValueError('entities outside finite alphabet')
        edges=[[a,b] for a in range(n) for b in range(n) if rng.random()<.2]
        return {'time':0.,'tick':0,'edges':edges,'query':[rng.randrange(n),rng.randrange(n)],'horizon':configuration.get('horizon',8)}
    def closure(self,s):
        rel=set(map(tuple,s['edges']))
        while True:
            more=rel|{(a,d) for a,b in rel for c,d in rel if b==c}
            if more==rel:return rel
            rel=more
    def advance(self,s,actions,dt,rng):
        reward=0.
        for a in actions:
            if a.verb=='assert':reward=float(a.arg('value')==(tuple(a.arg('pair')) in self.closure(s)))
        s['done']=s['tick']+1>=s['horizon'];return s,{}, {'correct':reward}
    def observe(self,s):
        closure=self.closure(s)
        return {'edges':Value.of(EDGES,map(tuple,s['edges'])),'query':Value.of(EDGE,tuple(s['query']))},{'closure':Value.of(EDGES,closure)},{'target':Value.of(BOOL,tuple(s['query']) in closure)},{'agent_0':('wait','assert')}
