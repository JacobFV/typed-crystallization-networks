"""Build coarse arithmetic/analytic regions entirely from universal operators."""
import random
from .types import floating, Value, product
from .operators import Registry
from .graph import Program,Node,Candidate
F=floating()

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
