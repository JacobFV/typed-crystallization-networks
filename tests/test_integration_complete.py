import json
import random
import torch
import pytest
from tcn.types import BOOL,Value,product
from tcn.generation import Host,Action,TimedValue,Generator
from tcn.graph import Program,Node,Candidate
from tcn.operators import Registry
from tcn.learning import SoftProgram
from tcn.policy import ActionBinding,parameter_width
from tcn.training import TrainConfig,Target,JointTrainer,target_loss
from tcn.scaffold import F,arithmetic_scaffold

def test_continuous_control_arguments_condition_prediction_and_exact_actor():
    torch.set_num_threads(1)
    host=Host.create('signal',configuration={'window':2,'components':1})
    binding=ActionBinding(0,'force',F,'force_parameters',(-2.,2.))
    inputs=(('samples',host.view().observations['samples'].type),('action',product(F)),(binding.input_port,F),('dt',F))
    p=arithmetic_scaffold(inputs,{'policy':1,'value':1,'prediction':3,'force_parameters':2},hidden=2)
    c=TrainConfig('signal',('samples',),(Target('probes','future'),),(Action('drive'),),{'window':2,'components':1},episodes=2,horizon=2,action_bindings=(binding,))
    t=JointTrainer(SoftProgram(p),c);history=t.run();assert all(x['gradient_norm']>0 for x in history)
    a=Action('drive',arguments=(('force',Value.of(F,1.5)),))
    assert t.inputs(host.view(),0,a)[binding.input_port].item()==1.5
    from tcn.agent import Agent
    agent=Agent(t.model.export(),t.model.registry,c);agent.rollout(host,steps=2,deterministic=True)
    assert len(host.inputs)==2;host.replay()

def test_distributional_targets_have_calibrating_gradients():
    pred=torch.tensor([.1,-.3],requires_grad=True)
    loss=target_loss(Target('probes','x',loss='gaussian'),pred,torch.tensor([2.]))
    loss.backward();assert pred.grad[0]<0 and torch.isfinite(pred.grad).all()
    b=torch.tensor([0.],requires_grad=True)
    target_loss(Target('probes','x',loss='bernoulli'),b,torch.tensor([1.])).backward();assert b.grad.item()<0

def test_packed_boolean_matches_exact_all_truth_tables_and_recurrence():
    from tcn.packed import PackedBoolean
    r=Registry();rng=random.Random(3)
    nodes=tuple(Node('n'+str(i),BOOL,(Candidate(r.resolve('truth_'+str(i),(BOOL,BOOL)),('a','b')),),selected=0) for i in range(16))
    p=Program((('a',BOOL),('b',BOOL)),nodes,tuple((n.name,n.name) for n in nodes))
    rows=[{k:Value.of(BOOL,bool(rng.randrange(2))) for k in ('a','b')} for _ in range(1024)]
    assert PackedBoolean(p).batch(rows)==[p.run(row)[0] for row in rows]
    p=Program((('a',BOOL),),(Node('n',BOOL,(Candidate(r.resolve('xor',(BOOL,BOOL)),('a','s')),),selected=0),),(('out','n'),),state=(('s',Value.of(BOOL,False),'n'),))
    packed=PackedBoolean(p);out,state=packed.run({'a':0b101},3);out,state=packed.run({'a':0b110},3,state);assert out['out']==0b011

def test_record_preserves_future_availability():
    class Delayed(Generator):
        def observe(self,s):return {'public/x':TimedValue(Value.of(BOOL,True),0.,5.)},{},{},{'agent_0':()}
    record=Delayed().record({'time':1.,'tick':1})
    assert record.actor_view().observations=={};assert record.observations['public/x'].available_at==5.

def test_checkpoint_keeps_frozen_flags_and_precision(tmp_path):
    from examples.joint import trainer
    t=trainer(episodes=1);t.model.freeze('bit_0');t.model.quantization['z']=(F,.3);t.run(tmp_path)
    restored=JointTrainer.load(tmp_path/'checkpoint.pt')
    assert restored.model.frozen==t.model.frozen
    assert restored.model.quantization==t.model.quantization
    assert [p.requires_grad for p in restored.model.parameters()]==[p.requires_grad for p in t.model.parameters()]

def parallel_runner(stage,out):
    import os
    out.mkdir(parents=True,exist_ok=True)
    return {'pid':os.getpid(),'success':1}

def test_curriculum_process_workers(tmp_path):
    import os
    from tcn.curriculum import Curriculum,Stage
    c=Curriculum([Stage(str(i),(),'test',{}, {'success':{'min':1}}) for i in range(3)])
    result=c.run(parallel_runner,tmp_path,workers=2)
    assert all(x['status']=='passed' and x['metrics']['pid']!=os.getpid() for x in result.values())

def test_all_executable_language_lessons_generate():
    from generators.language.engine.registry import all_lessons
    for lesson in all_lessons(implemented_only=True).values():
        for seed in (0,37):
            example=lesson.example(seed=seed,language='english')
            assert example.prompt and example.answer

def test_occluded_focus_is_not_visible():
    from generators.world_3d.generator import Implementation
    g=Implementation()
    h=Host.create('world_3d',configuration={'objects':[
        {'id':'paper','kind':'paper','mesh':'box','static':True,'position':[0.,.6,-1.],'size':[.4,.4,.03],'text':'secret'},
        {'id':'wall','kind':'object','mesh':'box','static':True,'position':[0.,.6,-.5],'size':[2.,2.,.1]}], 'reach':3.})
    assert not g.line_of_sight(h.state,'agent_0','paper')
    h.step((Action('read','paper'),),.01)
    assert h.state['agents']['agent_0']['result']=='occluded'
    assert h.state['agents']['agent_0']['focus'] is None
