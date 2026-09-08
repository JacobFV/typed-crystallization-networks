"""Learn logic -> explicit conversion -> arithmetic -> analytic composition."""
import math
from tcn.types import BOOL,Value,floating
from tcn.operators import Registry
from tcn.graph import Program,Node,Candidate,Signal
F=floating()

def problem():
    r=Registry()
    def candidates(names,types,sources,output=None):return tuple(Candidate(r.resolve(n,types,output),sources) for n in names)
    nodes=(
        Node('logic',BOOL,candidates([f'truth_{i}' for i in range(16)],(BOOL,BOOL),('a','b')),'logic',1),
        Node('conversion',F,candidates(['encode'],(BOOL,),('logic',),F),'encoding',2),
        Node('algebra',F,candidates(['add','sub','mul'],(F,F),('conversion','x')),'algebra',3),
        Node('analytic',F,candidates(['sin','identity'],(F,),('algebra',)),'readout',4))
    p=Program((('a',BOOL),('b',BOOL),('x',F)),nodes,(('answer','analytic'),),input_depths=(('x',2),))
    signals=(Signal('logic','logic',('logic',),BOOL,'bce'),Signal('conversion','conversion',('encoding',),F),Signal('algebra','algebra',('algebra',),F),Signal('analytic','answer',('readout',),F))
    examples=[]
    for a in [False,True]:
        for b in [False,True]:
            for x in [-.7,-.2,.3,.8]:
                z=float(a!=b);u=z+x
                examples.append({'inputs':{'a':Value.of(BOOL,a),'b':Value.of(BOOL,b),'x':Value.of(F,x)},'targets':{'logic':Value.of(BOOL,a!=b),'conversion':Value.of(F,z),'algebra':Value.of(F,u),'answer':Value.of(F,math.sin(u))}})
    return p,signals,examples
