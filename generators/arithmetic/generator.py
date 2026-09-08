from tcn.generation import Generator, SCALAR, Value, vector_value
from tcn.types import integer
I=integer(16)

class Implementation(Generator):
    action_schema={'wait':{},'answer':{'value':I}}
    def initialize(self,address,configuration):
        rng=address.rng(); objective=configuration.get('objective',{})
        operation=objective.get('operation',rng.choice(['add','sub','max']))
        if operation not in {'add','sub','max'}:raise ValueError('unknown objective operation')
        return {'time':0.,'tick':0,'x':rng.randrange(-4,5),'y':rng.randrange(-4,5),'operation':operation,'horizon':configuration.get('horizon',16)}
    def advance(self,s,actions,dt,rng):
        target=self.target(s); reward=0.
        for a in actions:
            if a.verb=='answer':reward=1. if a.arg('value')==target else -abs(a.arg('value')-target)/16
        s['x']=rng.randrange(-4,5);s['y']=rng.randrange(-4,5);s['done']=s['tick']+1>=s['horizon']
        return s,{'previous_target':Value.of(I,target)},{'correct':reward}
    def target(self,s):return {'add':lambda:s['x']+s['y'],'sub':lambda:s['x']-s['y'],'max':lambda:max(s['x'],s['y'])}[s['operation']]()
    def observe(self,s):
        return {'x':Value.of(I,s['x']),'y':Value.of(I,s['y']),'goal':vector_value([float(s['operation']==x) for x in ['add','sub','max']])},{'result':Value.of(I,self.target(s))},{'target':Value.of(I,self.target(s))},{'agent_0':('wait','answer')}
