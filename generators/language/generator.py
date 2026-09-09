"""Local symbolic construction and grammatical realization under the common contract."""
from .engine.registry import get,lesson_ids
from .engine.context import HARDENING,DEFAULT_HARDENING,NO_HARDENING
from tcn.generation import Generator,Value,BOOL,integer,product,text_value,read_text,vector_value
TEXT=text_value('',512).type

class Implementation(Generator):
    action_schema={'wait':{},'answer':{'text':TEXT}}
    def initialize(self,address,configuration):
        lesson=configuration.get('lesson','unification');language=configuration.get('language','english')
        # `difficulty` widens a lesson's own axes; `hardening` selects which
        # anti-exploit draws are in force. Both are versioned semantic
        # configuration, not a trainer branch: the actor sees only `text`.
        ex=get(lesson).example(seed=address.rng().randrange(2**31),language=language,
                               difficulty=configuration.get('difficulty'),
                               hardening=configuration.get('hardening'))
        data=ex.to_dict()
        return {'time':0.,'tick':0,'example':data,'response':'','capacity':configuration.get('capacity',1024),'horizon':configuration.get('horizon',4)}
    def advance(self,s,actions,dt,rng):
        reward=0.
        for a in actions:
            if a.verb=='answer':
                s['response']=read_text(dict(a.arguments)['text']);reward=float(s['response'].strip().casefold()==s['example']['answer'].strip().casefold())
        s['done']=s['tick']+1>=s['horizon'];return s,{}, {'correct':reward}
    def observe(self,s):
        ex=s['example'];hidden=ex.get('metadata',{}).get('hidden',{})
        # Construction remains privileged. Structured facts are emitted recursively
        # as products, sets, and encoded strings, never forwarded to the actor.
        def typed(x):
            if isinstance(x,bool):return Value.of(BOOL,x)
            if isinstance(x,int):return Value.of(integer(64),x)
            if isinstance(x,float):
                from tcn.generation import SCALAR
                return Value.of(SCALAR,x)
            if isinstance(x,dict):
                vals=[(text_value(str(k),128),typed(v)) for k,v in sorted(x.items())]
                fields=[Value(product(a.type,b.type),(a.raw,b.raw)) for a,b in vals]
            elif isinstance(x,(list,tuple)):fields=[typed(v) for v in x]
            else:return text_value(str(x),max(64,len(str(x).encode('utf8'))))
            return Value(product(*(v.type for v in fields)),tuple(v.raw for v in fields))
        return {'text':text_value(ex['prompt'],s['capacity'])},{'construction':typed(hidden)},{'answer':text_value(ex['answer'],512)},{'agent_0':('wait','answer')}
