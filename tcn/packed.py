"""Bit-parallel backend for crystallized Boolean programs, across batch lanes.

This lowers existing operators, introducing no synthesis primitives. Mixed-type
programs continue to use the exact general executor.
"""
from .types import BOOL,Value

class PackedBoolean:
    def __init__(self,program):
        program.validate()
        if not program.is_frozen:raise ValueError('packed execution requires frozen choices')
        if any(t!=BOOL for t in program.port_types().values()):raise TypeError('packed backend requires Boolean ports')
        self.program=program;self.instructions=[]
        for n in program.nodes:
            c=n.candidates[n.selected];name=c.operator.name
            if name not in {'identity','not','and','or','xor','nand','nor','xnor','mux','eq'} and not name.startswith('truth_'):raise TypeError('unsupported packed operator '+name)
            self.instructions.append((n.name,name,c.sources))
    def run(self,inputs,lanes,state=None):
        if lanes<1:raise ValueError('positive lane count required')
        mask=(1<<lanes)-1
        if set(inputs)!=set(dict(self.program.inputs)):raise ValueError('input ports mismatch')
        if any(not isinstance(x,int) or x<0 or x>mask for x in inputs.values()):raise ValueError('invalid packed input')
        v=dict(inputs)
        v.update({k:mask if x.decoded else 0 for k,x in self.program.constants})
        if state is not None and (set(state)!={k for k,_,_ in self.program.state} or any(not isinstance(x,int) or x<0 or x>mask for x in state.values())):raise ValueError('invalid packed state')
        v.update({k:mask if x.decoded else 0 for k,x,_ in self.program.state} if state is None else state)
        for dest,name,sources in self.instructions:
            xs=[v[k] for k in sources];a=xs[0];b=xs[1] if len(xs)>1 else 0
            if name=='identity':y=a
            elif name=='not':y=~a
            elif name=='mux':y=(a&b)|(~a&xs[2])
            elif name.startswith('truth_'):
                table=int(name[6:]);terms=(~a&~b,~a&b,a&~b,a&b);y=0
                for i,term in enumerate(terms):
                    if (table>>i)&1:y|=term
            else:
                y={'and':lambda:a&b,'or':lambda:a|b,'xor':lambda:a^b,'nand':lambda:~(a&b),'nor':lambda:~(a|b),'xnor':lambda:~(a^b),'eq':lambda:~(a^b)}[name]()
            v[dest]=y&mask
        return {k:v[s] for k,s in self.program.outputs},{k:v[s] for k,_,s in self.program.state}
    def batch(self,examples):
        if not examples:return []
        inputs={k:sum(int(row[k].decoded)<<i for i,row in enumerate(examples)) for k,_ in self.program.inputs}
        out,_=self.run(inputs,len(examples))
        return [{k:Value.of(BOOL,bool((bits>>i)&1)) for k,bits in out.items()} for i in range(len(examples))]
