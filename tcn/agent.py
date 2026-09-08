"""One frozen-agent loop for all generator domains."""
import math
import random
from .generation import Host
from .policy import bind_action,action_inputs
from .types import Value,floating,product

class Agent:
    def __init__(self,program,registry,configuration,seed=0):
        if not program.is_frozen:raise ValueError('agent runtime requires crystallized program')
        self.program=program;self.registry=registry;self.configuration=configuration;self.rng=random.Random(seed);self.state=None;self.previous=0;self.executed=None
    def reset(self):self.state=None;self.previous=0;self.executed=None
    def act(self,view,deterministic=False):
        c=self.configuration;inputs={k:view.observations[k] for k in c.observations}
        types=dict(self.program.inputs)
        if 'action' in types:inputs['action']=Value.of(types['action'],tuple(float(i==self.previous) for i in range(len(c.action_templates))))
        if 'dt' in types:inputs['dt']=Value.of(types['dt'],c.dt)
        inputs.update(action_inputs(c.action_bindings,types,self.previous,self.executed))
        outputs,state=self.program.run(inputs,self.state,self.registry)
        logits=outputs['policy'].flat();allowed=[i for i,a in enumerate(c.action_templates) if a.verb in view.available_actions]
        if not allowed:raise ValueError('no available action candidate')
        if deterministic:i=max(allowed,key=lambda i:logits[i])
        else:
            largest=max(logits[i] for i in allowed);weights=[math.exp(logits[i]-largest) for i in allowed];i=self.rng.choices(allowed,weights=weights,k=1)[0]
        action,_,_=bind_action(c.action_templates[i],i,c.action_bindings,outputs,training=False,rng=self.rng,deterministic=deterministic)
        inputs.update(action_inputs(c.action_bindings,types,i,action))
        if 'action' in types:inputs['action']=Value.of(types['action'],tuple(float(j==i) for j in range(len(c.action_templates))))
        _,self.state=self.program.run(inputs,self.state,self.registry)
        self.previous=i;self.executed=action;return action
    def rollout(self,host,steps=None,deterministic=False,actor='agent_0'):
        self.reset()
        for _ in range(steps or self.configuration.horizon):
            if host.records[-1].done:break
            from dataclasses import replace
            host.step((replace(self.act(host.view(actor),deterministic),actor=actor),),self.configuration.dt)
        return host
