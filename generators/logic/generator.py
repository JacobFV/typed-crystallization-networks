from tcn.generation import Generator, BOOL, Value, Address, vector_value
from tcn.types import product

# A two-input table is degenerate when its output ignores at least one input:
# the two constants, both projections, and both negated projections. Circuits
# drawn from the full 16 collapse to low-arity functions as depth grows, so
# `depth` alone is not a difficulty axis. `nondegenerate` restricts the pool to
# the ten tables that depend on both inputs; `min_relevant_inputs` constrains
# the sampled circuit itself.
DEGENERATE = (0, 3, 5, 10, 12, 15)
NONDEGENERATE = tuple(t for t in range(16) if t not in DEGENERATE)

def evaluate(width, gates, assignment):
    """Exact final value of a gate list under one input assignment."""
    values = [bool((assignment >> i) & 1) for i in range(width)]
    for a, b, table in gates:
        values.append(bool((table >> (2 * int(values[a]) + int(values[b]))) & 1))
    return values[-1]

def relevant_inputs(width, gates):
    """Input indices the final value actually depends on, by exact sensitivity."""
    relevant = set()
    for assignment in range(1 << width):
        base = evaluate(width, gates, assignment)
        for i in range(width):
            if i not in relevant and evaluate(width, gates, assignment ^ (1 << i)) != base:
                relevant.add(i)
        if len(relevant) == width:
            break
    return relevant

class Implementation(Generator):
    action_schema={"wait":{},"answer":{"value":BOOL}}
    def draw(self,rng,configuration,width,count):
        """One seeded circuit draw. The default pool preserves the original stream."""
        pool=configuration.get('tables')
        if pool is None and configuration.get('nondegenerate'):pool=NONDEGENERATE
        bits=[bool(rng.randrange(2)) for _ in range(width)];gates=[];values=bits[:]
        for _ in range(count):
            a=0 if configuration.get('fixed_inputs') else rng.randrange(len(values))
            b=1 if configuration.get('fixed_inputs') else rng.randrange(len(values))
            # An explicit `table` always wins: a pool restricts what is *drawn*, it
            # does not override what the caller asked for. Ignoring it silently made
            # a held-out-table experiment evaluate on the training distribution.
            # The no-pool branch keeps `configuration.get('table', rng.randrange(16))`
            # verbatim: the argument is evaluated either way, so the stream advances
            # even when a fixed table is supplied. Recorded episodes depend on that.
            if pool is None: table=int(configuration.get('table',rng.randrange(16)))
            elif 'table' in configuration: table=int(configuration['table'])
            else: table=int(pool[rng.randrange(len(pool))])
            if not 0<=table<=15: raise ValueError('truth table must be 0..15')
            gates.append([a,b,table]);values.append(bool((table>>(2*int(values[a])+int(values[b])))&1))
        return bits,gates,values
    def initialize(self,address,configuration):
        rng=address.rng(); count=int(configuration.get('depth',3)); width=int(configuration.get('inputs',4))
        if not 1<=count<=64: raise ValueError('depth must be 1..64')
        if not 1<=width<=16: raise ValueError('inputs must be 1..16')
        if configuration.get('fixed_inputs') and width<2: raise ValueError('fixed_inputs requires at least two inputs')
        minimum=int(configuration.get('min_relevant_inputs',0))
        if not 0<=minimum<=width: raise ValueError('min_relevant_inputs must be 0..inputs')
        if minimum:
            # Rejection sampling on the seeded stream, so replay is unaffected.
            if width>12: raise ValueError('min_relevant_inputs requires inputs <= 12')
            attempts=int(configuration.get('max_attempts',4096))
            if attempts<1: raise ValueError('max_attempts must be positive')
            best=-1
            for _ in range(attempts):
                bits,gates,values=self.draw(rng,configuration,width,count)
                best=max(best,len(relevant_inputs(width,gates)))
                if best>=minimum: break
            else:
                # High-arity targets are rare at shallow depth: a function of all
                # w inputs needs at least w-1 two-input gates, and the sampled
                # wiring must supply them. Report the shortfall rather than a
                # bare failure, since it is a property of the request.
                raise ValueError(f'no sampled circuit reached min_relevant_inputs={minimum} in {attempts} draws at depth={count}, inputs={width}; best was {best}. Increase depth or max_attempts.')
        else: bits,gates,values=self.draw(rng,configuration,width,count)
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
