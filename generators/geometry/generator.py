from tcn.generation import Generator,SCALAR,Value,image_value,vector_value
from tcn.types import product
from .render import render
VEC3=product(SCALAR,SCALAR,SCALAR)

class Implementation(Generator):
    action_schema={'wait':{},'camera':{'delta':VEC3},'translate':{'delta':VEC3},'rotate':{'angles':VEC3}}
    def initialize(self,address,configuration):
        rng=address.rng();n=int(configuration.get('objects',3))
        if not 1<=n<=32:raise ValueError('object count outside capacity')
        objects=[{'id':f'body_{i}','mesh':rng.choice(['box','sphere','cylinder']),'position':[rng.uniform(-1,1),rng.uniform(.3,1.5),rng.uniform(-1,1)],'size':[rng.uniform(.3,.8) for _ in range(3)],'rotation':[rng.uniform(-.3,.3) for _ in range(3)],'color':[rng.randrange(40,240) for _ in range(3)]} for i in range(n)]
        return {'time':0.,'tick':0,'objects':objects,'camera':{'eye':[3.,3.,5.],'target':[0.,.5,0.]},'resolution':configuration.get('resolution',32),'horizon':configuration.get('horizon',8)}
    def advance(self,s,actions,dt,rng):
        for a in actions:
            if a.verb=='camera':s['camera']['eye']=[x+y for x,y in zip(s['camera']['eye'],a.arg('delta'))]
            if a.verb in {'translate','rotate'}:
                obj=next((o for o in s['objects'] if o['id']==a.target),None)
                if obj is None:raise ValueError('unknown target')
                field='position' if a.verb=='translate' else 'rotation';key='delta' if a.verb=='translate' else 'angles'
                obj[field]=[x+y for x,y in zip(obj[field],a.arg(key))]
        s['done']=s['tick']+1>=s['horizon'];return s,{},{}
    def observe(self,s):
        rgb,depth,ids,normals=render(s['objects'],s['camera'],s['resolution'],s['resolution'])
        return {'pixels':image_value(rgb)},{'transforms':vector_value([x for o in s['objects'] for x in o['position']+o['rotation']])},{'depth':vector_value(depth.flatten()),'object_ids':vector_value(ids.flatten()),'normals':vector_value(normals.flatten())},{'agent_0':tuple(self.action_schema)}
