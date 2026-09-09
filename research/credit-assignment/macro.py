"""Composite actions as crystallized modules, and the test that they are not an oracle menu.

ARCHITECTURE.md section 4 makes a crystallized sub-graph an immutable, versioned
callable operator, charged to description size and execution cost. Section 5 says a
frozen module has no internal gradients. Applying that to the *action-emitting*
region rather than the perception region is what a composite action is here: a
sub-program from the executed-action ports to a primitive action's typed argument,
frozen, registered in the `Registry`, and offered to the outer policy as one more
candidate at one node.

This script builds two such modules from crystallized sub-graphs -- `sweep`
(`add(previous slot, 1)`, the sub-action that solves the task) and `stare`
(`identity(previous slot)`, the one that does not) -- installs them as the only two
candidates of `next_slot`, and trains the outer policy from reward alone. The
question it answers is the one the brief asks: can the agent *discover* which
sub-action is meaningful, rather than being handed the useful one?

Why this is not an oracle action menu (AGENTS.md forbids one as a model input):

1. The environment's menu is unchanged. `StepRecord.available_actions` is
   `('wait','look','dial','commit')` at every tick of every panel episode, and
   `ActorView` carries nothing else. An oracle menu is environment-side and
   state-dependent -- it tells the agent which actions are useful *here*. This
   library is agent-side, fixed for the episode, and says nothing about the state.
2. The library contains a wrong member, and which is which is not marked. If the
   outer policy could read usefulness off the menu, the `stare` arm below would not
   be measurable; it is, and it is worse.
3. It is charged for. `Program.description_bits` charges each distinct module
   definition once and every call site individually, so a macro that does not pay
   for itself loses under `L_program_description`. An oracle menu is free.
4. Nothing was added to the algebra. `Registry.register_module` already makes a
   frozen program a callable operator with `gradient="none"`, so the *choice among*
   macros is a candidate softmax like any other while the macro's internals are a
   hard gradient boundary -- which is exactly the abstraction boundary section 4
   asks for.

Run: .venv/bin/python research/credit-assignment/macro.py [episodes]
"""
import json,statistics,sys,time
import torch
ROOT='/home/brandonin/Documents/typed-crystallization-networks'
sys.path.insert(0,ROOT);sys.path.insert(0,ROOT+'/research/credit-assignment')
from tcn.graph import Program,Node,Candidate
from tcn.operators import Registry
from tcn.types import Value
from panel import Counter
from generators.computer.generator import SLOT
import program as PR
from run import Runner,Cfg,SHARP


def sub_action(registry,rule):
    """One crystallized sub-action: executed slot -> next slot, frozen."""
    constants=(('one_slot',Value.of(SLOT,1)),)
    if rule=='sweep':
        candidate=Candidate(registry.resolve('add',(SLOT,SLOT)),('action.1.slot','one_slot'))
    else:
        candidate=Candidate(registry.resolve('identity',(SLOT,)),('action.1.slot',))
    node=Node('next_slot',SLOT,(candidate,),'plan',1,0)
    return Program((('action.1.slot',SLOT),),(node,),(('next_slot','next_slot'),),constants).validate(registry)


def macro_program(library,**kw):
    """`panel_program` with `next_slot` replaced by a choice over module call sites."""
    base,registry=PR.panel_program(**kw)
    names=[registry.register_module(sub_action(registry,rule)) for rule in library]
    candidates=tuple(Candidate(registry.resolve(name,(SLOT,),SLOT),('action.1.slot',)) for name in names)
    nodes=tuple(Node(n.name,n.output,candidates,n.region,n.depth,None) if n.name=='next_slot' else n
                for n in base.nodes)
    program=Program(base.inputs,nodes,base.outputs,base.constants,
                    trainable_constants=base.trainable_constants).validate(registry)
    return program,registry,names


class MacroRunner(Runner):
    """`Runner` with the module library installed at `next_slot`."""
    def __init__(self,cfg,counter,library):
        self.library=library
        from tcn.learning import SoftProgram
        self.cfg=cfg;self.counter=counter
        torch.manual_seed(cfg.seed);torch.set_num_threads(1)
        program,registry,names=macro_program(library,perception=cfg.perception,
                                             policy_state=cfg.policy_state,seed_logits=cfg.seed_logits)
        self.program=program;self.registry=registry;self.module_names=names
        self.model=SoftProgram(program,registry)
        for name in SHARP:self.model.temperatures[name]=cfg.sharp
        params=[p for p in self.model.choices if p.requires_grad]+list(self.model.constants.parameters())
        for p in self.model.parameters():p.requires_grad_(any(p is q for q in params))
        self.params=params
        self.optimizer=torch.optim.Adam(params,lr=cfg.lr)
        self.history=[];self.types=dict(program.inputs)

    def candidate_names(self):
        out={}
        for node,p in zip(self.program.nodes,self.model.choices):
            if len(node.candidates)>1:
                i=int(torch.argmax(p))
                out[node.name]=self.library[i] if node.name=='next_slot' else node.candidates[i].operator.name
        return out


def costs(library):
    """Description bits and execution cost of the hardened program, per library."""
    program,registry,names=macro_program(library)
    rows={}
    for i,rule in enumerate(library):
        selections={n.name:(i if n.name=='next_slot' else 0) for n in program.nodes}
        chosen=program.harden(selections).pruned()
        rows[rule]={'description_bits':chosen.description_bits(registry),
                    'execution_cost':chosen.execution_cost(registry)}
    inlined,r2=PR.panel_program()
    sel={n.name:(0 if n.name!='next_slot' else
                 [f'{c.operator.name}({",".join(c.sources)})' for c in n.candidates].index('add(action.1.slot,one_slot)'))
         for n in inlined.nodes}
    hardened=inlined.harden(sel).pruned()
    rows['inlined_sweep']={'description_bits':hardened.description_bits(r2),
                           'execution_cost':hardened.execution_cost(r2)}
    return rows


def main(episodes=400,seed=0,library=('sweep','stare')):
    t0=time.perf_counter()
    counter=Counter(session=f'macro{"_".join(library)}{seed}')
    runner=MacroRunner(Cfg(episodes=episodes,seed=seed),counter,library)
    runner.train(log_every=100)
    report={'library':list(library),'seed':seed,'episodes':episodes,
            'final':runner.evaluate(64),'final_stochastic':runner.evaluate(64,stochastic=True),
            'selected':runner.candidate_names().get('next_slot',library[0]),
            'logits':runner.report()['logits'],'history':runner.history,
            'environment_episodes':counter.episodes,'seconds':time.perf_counter()-t0}
    if seed==0 and len(library)>1:report['costs']=costs(library)
    name='_'.join(library)
    open(ROOT+f'/research/credit-assignment/out/macro_{name}_s{seed}.json','w').write(json.dumps(report,indent=2,default=str))
    print(json.dumps({k:report[k] for k in ('library','seed','final','selected','environment_episodes','seconds')},default=str))

if __name__=='__main__':
    episodes=int(sys.argv[1]) if len(sys.argv)>1 else 400
    seed=int(sys.argv[2]) if len(sys.argv)>2 else 0
    library=tuple(sys.argv[3].split(',')) if len(sys.argv)>3 else ('sweep','stare')
    main(episodes,seed,library)
