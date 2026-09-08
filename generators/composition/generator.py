"""Declarative composition of any registered generators, with typed action links."""
from tcn.generation import Generator,Host,Action,load_generator

class Implementation(Generator):
    def configure(self,configuration):
        self.action_schema={}
        for name,spec in configuration.get('children',{}).items():
            child=load_generator(spec['generator']);child.configure(spec.get('configuration',{}))
            self.action_schema.update({name+'::'+verb:args for verb,args in child.action_schema.items()})
        self.action_schema['wait']={}
    def initialize(self,address,configuration):
        self.configure(configuration)
        specs=configuration.get('children',{})
        if not specs:raise ValueError('composition requires children')
        children={}
        for i,(name,spec) in enumerate(specs.items()):
            children[name]=Host.create(spec['generator'],seed=address.rng(name).randrange(2**31),configuration=spec.get('configuration',{}),objective=spec.get('objective',{})).snapshot()
        return {'time':0.,'tick':0,'children':children,'links':configuration.get('links',[]),'horizon':configuration.get('horizon',8)}
    def advance(self,s,actions,dt,rng):
        grouped={k:[] for k in s['children']}
        for a in actions:
            if a.verb=='wait':continue
            child,verb=a.verb.split('::',1);grouped[child].append(Action(verb,a.target,a.arguments,a.actor))
        # All links read the pre-transition records; cycles therefore have a delay.
        for link in s['links']:
            source=Host.restore(s['children'][link['source']]).records[-1]
            channel=link.get('channel','observations')
            if channel not in {'observations','probes','latent_states'}:raise ValueError('invalid link channel')
            value=getattr(source,channel)[link['port']]
            if channel=='observations':value=value.value
            grouped[link['target']].append(Action(link['verb'],link.get('object',''),((link['argument'],value),),link.get('actor','agent_0')))
        rewards={};transitions={}
        for name,snap in s['children'].items():
            h=Host.restore(snap)
            if not h.records[-1].done:
                record=h.step(grouped[name],dt,rng.randrange(2**31));s['children'][name]=h.snapshot()
                rewards.update({name+'.'+k:v.decoded for k,v in record.reward_components.items()})
                transitions.update({name+'.'+k:v for k,v in record.transition.items()})
        s['done']=s['tick']+1>=s['horizon'] or all(Host.restore(x).records[-1].done for x in s['children'].values())
        return s,transitions,rewards
    def observe(self,s):
        observations={};latents={};probes={};menus={}
        for name,snap in s['children'].items():
            record=Host.restore(snap).records[-1]
            for k,v in record.observations.items():
                owner,sep,key=k.partition('/')
                if not sep:owner='public';key=k
                observations[owner+'/'+name+'.'+key]=v
            latents.update({name+'.'+k:v for k,v in record.latent_states.items()});probes.update({name+'.'+k:v for k,v in record.probes.items()})
            for actor,verbs in record.available_actions.items():menus[actor]=menus.get(actor,())+tuple(name+'::'+v for v in verbs)
        return observations,latents,probes,menus
