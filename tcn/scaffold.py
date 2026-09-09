"""Build coarse arithmetic/analytic regions entirely from universal operators."""
import random
from .types import floating, integer, setof, Value, product
from .operators import Registry
from .graph import Program,Node,Candidate
F=floating()

def positional_scaffold(registry,observation,positions,modules,index=None,region='core',port='observation'):
    """Apply one crystallized module at many positions of a wide tuple.

    `map` and `filter` need a `set`, and perceptual observations are `tuple`s, so
    a wide input has to become an iterable of positions before a shared module
    can run over it. It does, using only registered operators:

        held    = insert(empty_set_constant, observation)   set[Observation], capacity 1
        records = pair(position_constant_set, held)         set[(Index, Observation)]
        mapped  = map(records; module = m)                  set[(Index, Out)]

    `pair` is the cartesian product, so `records` carries one record per position
    with the whole observation attached, and the module reads its own position
    out of the record and indexes there. Three nodes for any number of positions,
    so the description charges the module definition once and the caller does not
    grow with the input width, while execution stays charged per position.

    This is a composition of existing operators, not a new one. Nothing here is
    specific to rasters: `observation` is any uniform tuple and `positions` are
    ordinary integer values, so the same scaffold serves a signal, a sequence of
    symbols, or a set of coordinates.

    Several `modules` give the node one candidate each, so which shared
    sub-program runs at every position is a learned or searched choice. They must
    agree on their interface, since a node has one output type.
    """
    if observation.kind!='tuple' or not observation.items:
        raise TypeError('positional reuse requires a nonempty tuple observation')
    positions=tuple(positions)
    if not positions: raise ValueError('at least one position required')
    if not modules: raise ValueError('at least one crystallized module required')
    index=index or integer(32,signed=False)
    if index.kind!='int' or index.encoding.kind!='integer' or index.role:
        raise TypeError('positions need a plain integer index encoding')
    holder=setof(observation,1);locations=setof(index,len(set(positions)))
    hold=registry.resolve('insert',(holder,observation))
    records=registry.resolve('pair',(locations,hold.output))
    ops=[registry.resolve('map',(records.output,),None,{'module':m}) for m in modules]
    if len({o.output for o in ops})!=1:
        raise TypeError('modules mapped at one node must share an output interface')
    nodes=(Node('held',hold.output,(Candidate(hold,('empty',port)),),region,1,0),
           Node('records',records.output,(Candidate(records,('positions','held')),),region,2,0),
           Node('mapped',ops[0].output,tuple(Candidate(o,('records',)) for o in ops),region,3,
                0 if len(ops)==1 else None))
    constants=(('positions',Value.of(locations,positions)),('empty',Value.of(holder,())))
    return Program(((port,observation),),nodes,(('mapped','mapped'),),constants,
                   input_depths=((port,0),)).validate(registry)

def arithmetic_scaffold(inputs,outputs,hidden=8,seed=0):
    """Explicit mul/add/sin graph; no opaque neural layer or domain preprocessing."""
    r=Registry();rng=random.Random(seed);nodes=[];constants=[];trainable=[];types=dict(inputs);depths={k:0 for k in types}
    def node(name,op,sources,region,output=None,alternatives=()):
        ops=[r.resolve(x,tuple(types[s] for s in sources),output) for x in (op,*alternatives)]
        n=Node(name,ops[0].output,tuple(Candidate(o,tuple(sources)) for o in ops),region,max([depths[s] for s in sources],default=0)+1)
        nodes.append(n);types[name]=n.output;depths[name]=n.depth;return name
    def parameter(name,value):
        constants.append((name,Value.of(F,value)));trainable.append(name);types[name]=F;depths[name]=-1;return name
    features=[]
    def flatten(ref,t):
        if t.kind=='tuple':
            for i,field in enumerate(t.items):
                name=f'field_{len(nodes)}';op=r.resolve('project',(t,),parameters={'index':i})
                n=Node(name,field,(Candidate(op,(ref,)),),'encoding',depths[ref]+1);nodes.append(n);types[name]=field;depths[name]=n.depth;flatten(name,field)
        elif t.kind in {'int','bool'}:
            if t==F:features.append(ref)
            else:
                # Explicit conversion node retains semantic role; numeric use is
                # allowed only for a scalar schema, never category IDs.
                if t.role in {'category','symbol'}:raise TypeError('supply explicit categorical encoding before this scaffold')
                out=F
                if t.unit or t.frame or t.role:raise TypeError('explicit semantic/unit normalization required')
                features.append(node(f'encoding_{len(nodes)}','decode',[ref],'encoding',out))
        else:raise TypeError('set scaffold requires an explicit set encoding program')
    for name,t in inputs:flatten(name,t)
    def linear(prefix,features,count,region):
        result=[]
        for j in range(count):
            terms=[]
            for i,ref in enumerate(features):
                w=parameter(f'{prefix}_w{j}_{i}',rng.gauss(0,1/max(1,len(features))**.5))
                terms.append(node(f'{prefix}_mul{j}_{i}','mul',[ref,w],region))
            bias=parameter(f'{prefix}_bias{j}',0.)
            packed=node(f'{prefix}_terms{j}','tuple',terms+[bias],region)
            result.append(node(f'{prefix}_sum{j}','sum',[packed],region))
        return result
    hiddenrefs=linear('hidden',features,hidden,'representation')
    hiddenrefs=[node(f'activation_{i}','sin',[ref],'representation',alternatives=('identity',)) for i,ref in enumerate(hiddenrefs)]
    outs=[]
    for name,count in outputs.items():
        refs=linear(name,hiddenrefs,count,name)
        packed=node(name+'_output','tuple',refs,name);outs.append((name,packed))
    return Program(tuple(inputs),tuple(nodes),tuple(outs),tuple(constants),trainable_constants=tuple(trainable)).validate(r)
