from tcn.generation import Generator, BOOL, Value, Address, vector_value
from tcn.types import product

class Implementation(Generator):
    action_schema={"wait":{},"answer":{"value":BOOL}}
    def initialize(self,address,configuration):
        rng=address.rng(); count=int(configuration.get('depth',3))
        if not 1<=count<=64: raise ValueError('depth must be 1..64')
        bits=[bool(rng.randrange(2)) for _ in range(4)]; gates=[]; values=bits[:]
        for _ in range(count):
            a=0 if configuration.get('fixed_inputs') else rng.randrange(len(values));b=1 if configuration.get('fixed_inputs') else rng.randrange(len(values));table=int(configuration.get('table',rng.randrange(16)))
            gates.append([a,b,table]);values.append(bool((table>>(2*int(values[a])+int(values[b])))&1))
        return {'time':0.,'tick':0,'bits':bits,'gates':gates,'values':values,'horizon':configuration.get('horizon',8),'answer':False,'invert':bool(configuration.get('objective',{}).get('invert',False))}
    def advance(self,state,actions,dt,rng):
        reward=0.
        for a in actions:
            if a.verb=='answer':state['answer']=a.arg('value');reward=float(state['answer']==(state['values'][-1]!=state['invert']))
        state['done']=state['tick']+1>=state['horizon']
        return state,{}, {'correct':reward}
    def observe(self,s):
        values=Value.of(product(*(BOOL for _ in s['values'])),tuple(s['values']))
        obs={'bits':Value.of(product(*(BOOL for _ in s['bits'])),tuple(s['bits'])),'program':vector_value([x for g in s['gates'] for x in g])}
        obs['goal']=Value.of(BOOL,s['invert'])
        target=Value.of(BOOL,s['values'][-1]!=s['invert'])
        return obs,{'values':values},{'target':target,'gate':Value.of(BOOL,s['values'][-1])},{'agent_0':('wait','answer')}
