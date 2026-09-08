import math
from tcn.generation import Generator, SCALAR, Value, vector_value

class Implementation(Generator):
    action_schema={'wait':{},'drive':{'force':SCALAR}}
    def initialize(self,address,configuration):
        rng=address.rng();n=int(configuration.get('components',3));window=int(configuration.get('window',16))
        if not 1<=n<=16 or not 2<=window<=256:raise ValueError('invalid signal capacity')
        return {'time':0.,'tick':0,'frequency':[rng.uniform(.02,.25) for _ in range(n)],'amplitude':[rng.uniform(.1,1) for _ in range(n)],'phase':[rng.uniform(-math.pi,math.pi) for _ in range(n)],'window':window,'samples':[0.]*window,'drive':0.,'horizon':configuration.get('horizon',64)}
    def value(self,s,t):return sum(a*math.sin(2*math.pi*f*t+p) for a,f,p in zip(s['amplitude'],s['frequency'],s['phase']))+s['drive']
    def advance(self,s,actions,dt,rng):
        for a in actions:
            if a.verb=='drive':s['drive']=a.arg('force')
        v=self.value(s,s['time']+dt);s['samples']=(s['samples']+[v])[-s['window']:];s['done']=s['tick']+1>=s['horizon']
        return s,{'sample':Value.of(SCALAR,v)},{'energy':-v*v}
    def observe(self,s):
        return {'samples':vector_value(s['samples']),'time':Value.of(SCALAR,s['time'])},{'frequencies':vector_value(s['frequency']),'phases':vector_value(s['phase'])},{'future':vector_value([self.value(s,s['time']+h) for h in [1,2,4]])},{'agent_0':('wait','drive')}
