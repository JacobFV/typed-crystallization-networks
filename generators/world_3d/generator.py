"""Embodied typed transition system with local observations and causal affordances."""
import copy
import math
from collections import Counter
import numpy as np
from tcn.generation import Generator,SCALAR,Value,BOOL,image_value,text_value,read_text,vector_value,integer
from tcn.types import product
from generators.geometry.render import render
from generators.raster_text.generator import render_text
from .physics import advance as physics_advance
VEC3=product(SCALAR,SCALAR,SCALAR);TEXT=text_value('',512).type

class Implementation(Generator):
    action_schema={'wait':{},'move':{'force':VEC3},'actuate':{'torque':SCALAR},'grasp':{},'release':{},'look':{'direction':VEC3},'read':{},'write':{'text':TEXT},'type':{'text':TEXT},'key':{'code':integer(16,signed=False)},'open':{},'press':{},'say':{'text':TEXT}}
    dimensions=3
    def initialize(self,address,configuration):
        rng=address.rng();n=int(configuration.get('agents',1));objects=copy.deepcopy(configuration.get('objects',[]))
        for i in range(n):objects.append({'id':f'agent_{i}','kind':'robot','mesh':'box','position':[i*1.2,.3,0.],'size':[.35,.5,.35],'mass':3.,'color':[60,150,220]})
        if not configuration.get('objects'):
            for i in range(int(configuration.get('bodies',3))):objects.append({'id':f'body_{i}','kind':'object','mesh':rng.choice(['box','sphere','cylinder']),'position':[rng.uniform(-1,1),.7,rng.uniform(-1,1) if self.dimensions==3 else 0.],'size':[.3,.3,.3],'color':[rng.randrange(80,230) for _ in range(3)]})
        ids=[o['id'] for o in objects]
        if len(set(ids))!=len(ids) or any(not x.replace('_','').isalnum() for x in ids):raise ValueError('invalid or duplicate body id')
        physical,poses,contacts=physics_advance(objects,None,.01,dimensions=self.dimensions)
        state={'time':0.,'tick':0,'objects':objects,'physical':physical,'poses':poses,'contacts':contacts,'agents':{f'agent_{i}':{'holding':None,'look':[0.,-.3,-1.],'messages':[],'result':'','focus':None} for i in range(n)},'reach':configuration.get('reach',1.2),'communication_range':configuration.get('communication_range',5.),'resolution':configuration.get('resolution',24),'horizon':configuration.get('horizon',64),'objective':configuration.get('objective',{}),'dimensions':self.dimensions,'computer':None}
        if any(o.get('kind')=='computer' for o in objects):
            from generators.computer.generator import Implementation as Computer
            state['computer']=Computer().initialize(address,{'interface':'keyboard','horizon':100000,'document':configuration.get('document','7')})
        return state
    def position(self,s,name):return np.array(s['poses'][name]['position'])
    def distance(self,s,a,b):return float(np.linalg.norm(self.position(s,a)-self.position(s,b)))
    def line_of_sight(self,s,actor,target):
        start=self.position(s,actor)+np.array([0,.3,.1]);end=self.position(s,target)
        for obj in s['objects']:
            if obj['id'] in {actor,target}:continue
            pose=s['poses'][obj['id']];rotation=np.array(pose['matrix'])
            origin=rotation.T@(start-np.array(pose['position']));direction=rotation.T@(end-start)
            half=np.array(obj.get('size',[.5,.5,.5]))/2;lo=0.;hi=1.
            for axis in range(3):
                if abs(direction[axis])<1e-10:
                    if abs(origin[axis])>half[axis]:lo=2.;break
                else:
                    a=(-half[axis]-origin[axis])/direction[axis];b=(half[axis]-origin[axis])/direction[axis]
                    lo=max(lo,min(a,b));hi=min(hi,max(a,b))
            if lo<hi and hi>1e-5 and lo<1-1e-5:return False
        return True
    def advance(self,s,actions,dt,rng):
        forces={};torques={};rewards={};counts=Counter(a.target for a in actions if a.verb in {'grasp','write','open','type','key'} and a.target)
        for a in sorted(actions,key=lambda a:(a.actor,a.verb,a.target)):
            if a.actor not in s['agents']:raise ValueError('unknown acting agent')
            agent=s['agents'][a.actor];agent['result']='ok'
            if a.verb=='move':forces[a.actor]=np.clip(a.arg('force'),-30,30).tolist()
            elif a.verb=='actuate':torques[a.actor]=a.arg('torque')
            elif a.verb=='look':agent['look']=list(a.arg('direction'))
            elif a.verb=='release':agent['holding']=None
            elif a.verb=='say':
                message=read_text(dict(a.arguments)['text'])
                for other in s['agents']:
                    if other!=a.actor and self.distance(s,a.actor,other)<=s['communication_range']:
                        s['agents'][other]['messages']=(s['agents'][other]['messages']+[message])[-4:]
            elif a.verb!='wait':
                obj=next((o for o in s['objects'] if o['id']==a.target),None)
                if obj is None or self.distance(s,a.actor,a.target)>s['reach']:agent['result']='out_of_reach';continue
                if not self.line_of_sight(s,a.actor,a.target):agent['result']='occluded';continue
                if counts[a.target]>1:agent['result']='conflict';continue
                if a.verb=='grasp':
                    if obj.get('static') or obj.get('kind')=='robot':agent['result']='not_graspable'
                    elif any(x['holding']==a.target for x in s['agents'].values()):agent['result']='occupied'
                    else:agent['holding']=a.target
                elif a.verb=='read':agent['focus']=a.target
                elif a.verb=='write':
                    if obj.get('kind')!='paper':agent['result']='not_writable'
                    else:obj['text']=read_text(dict(a.arguments)['text'])
                elif a.verb=='open':
                    if obj.get('kind')!='door':agent['result']='not_openable'
                    elif obj.get('locked',False):agent['result']='locked'
                    else:obj['opened']=True
                elif a.verb=='press':
                    if obj.get('kind')!='button':agent['result']='not_pressable'
                    else:
                        target=next((o for o in s['objects'] if o['id']==obj.get('controls')),None)
                        if target and target.get('kind')=='door':target['opened']=not target.get('opened',False)
                        else:agent['result']='unconnected'
                elif a.verb in {'type','key'}:
                    if obj.get('kind')!='computer':agent['result']='not_a_computer'
                    else:
                        from generators.computer.generator import Implementation as Computer
                        from tcn.generation import Action
                        cs,_,_=Computer().advance(s['computer'],[Action(a.verb,arguments=a.arguments)],dt,rng)
                        cs['time']=s['time']+dt;cs['tick']+=1;s['computer']=cs;agent['focus']=a.target
        for obj in s['objects']:
            if obj.get('kind')=='door':torques[obj['id']]=1.4 if obj.get('opened') else 0.
        s['physical'],s['poses'],s['contacts']=physics_advance(s['objects'],s['physical'],dt,forces,torques,{k:v['holding'] for k,v in s['agents'].items()},s['dimensions'])
        g=s['objective']
        if g.get('kind')=='reach':rewards['goal']=-self.distance(s,g.get('actor','agent_0'),g['target'])
        if g.get('kind')=='write':
            obj=next(o for o in s['objects'] if o['id']==g['target']);rewards['goal']=float(obj.get('text')==g['text'])
        rewards['effort']=-sum(sum(x*x for x in f) for f in forces.values())*.0001
        s['done']=s['tick']+1>=s['horizon'];return s,{'contacts':Value.of(integer(32,signed=False),len(s['contacts']))},rewards
    def observe(self,s):
        obs={};probes={};menus={};objects=[]
        for o in s['objects']:
            v=dict(o)|s['poses'][o['id']]
            if o.get('kind')=='computer' and s['computer']:
                out=s['computer']['result']['output'];v['text']=out.get('stdout',out.get('content',''))+s['computer']['buffer']
            objects.append(v)
            if o.get('kind')=='robot':objects.append({'id':'arm_'+o['id'],'mesh':'box','size':[.3,.08,.08],'color':[180,190,200]}|s['poses']['arm_'+o['id']])
        for name,agent in s['agents'].items():
            pos=self.position(s,name);eye=pos+np.array([0,.3,.1]);target=eye+np.array(agent['look'])
            visible=[o for o in objects if o['id']!=name]
            rgb,depth,ids,normals=render(visible,{'eye':eye.tolist(),'target':target.tolist()},s['resolution'],s['resolution'])
            obs[name+'/pixels']=image_value(rgb)
            obs[name+'/proprioception']=vector_value(list(pos)+s['poses'][name]['velocity'])
            obs[name+'/result']=text_value(agent['result'],64)
            obs[name+'/messages']=text_value('\n'.join(agent['messages']),2048)
            focus=next((o for o in objects if o['id']==agent['focus']),None)
            obs[name+'/focus_pixels']=image_value(render_text('',128,48,14))
            if focus and self.distance(s,name,focus['id'])<=s['reach'] and self.line_of_sight(s,name,focus['id']):
                if focus.get('kind')=='computer':
                    from generators.computer.generator import Implementation as Computer
                    obs[name+'/focus_pixels']=Computer().observe(s['computer'])[0]['pixels']
                elif focus.get('kind')=='paper':obs[name+'/focus_pixels']=image_value(render_text(focus.get('text',''),128,48,14))
            probes[name+'/depth']=vector_value(depth.flatten());probes[name+'/visible_ids']=vector_value(ids.flatten())
            menus[name]=tuple(self.action_schema)
        latents={'positions':vector_value([x for o in s['objects'] for x in s['poses'][o['id']]['position']]),'contacts':Value.of(integer(32,signed=False),len(s['contacts']))}
        return obs,latents,probes,menus
