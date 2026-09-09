import copy
import pytest
import torch
from tcn.curriculum import Curriculum,Stage
from tcn.training import JointTrainer

def test_curriculum_gates_resume_and_cycle(tmp_path):
    with pytest.raises(ValueError):Curriculum([Stage('a',('b',),'x',{},{}),Stage('b',('a',),'x',{}, {})])
    c=Curriculum([Stage('a',(),'x',{}, {'success':{'min':1}}),Stage('b',('a',),'x',{}, {})])
    calls=[]
    def run(s,p,artifacts):calls.append(s.name);return {'success':1}
    first=c.run(run,tmp_path,workers=1);assert all(v['status']=='passed' for v in first.values())
    c.run(run,tmp_path,workers=1);assert calls==['a','b']
    failed=c.run(lambda s,p,a:{'success':0},tmp_path/'failed');assert failed['a']['status']=='failed';assert failed['b']['status']=='blocked'

def test_joint_objectives_and_checkpoint_resume(tmp_path):
    from examples.joint import trainer
    torch.set_num_threads(1);t=trainer(episodes=4);before=copy.deepcopy(t.model.state_dict());history=t.run(tmp_path)
    assert len(history)==4;assert all(x['gradient_norm']>0 for x in history)
    assert all('prediction_loss' in x and 'policy_loss' in x for x in history)
    assert len({str(x['goal']) for x in history})==2
    assert any(not torch.equal(v,t.model.state_dict()[k]) for k,v in before.items())
    restored=JointTrainer.load(tmp_path/'checkpoint.pt')
    assert restored.completed==4;assert restored.history==history
    for key,value in t.model.state_dict().items():assert torch.equal(value,restored.model.state_dict()[key])
    restored.config.episodes=5;assert len(restored.run())==5


def test_parameter_and_choice_gradients_and_export():
    from tcn.scaffold import arithmetic_scaffold,F
    from tcn.learning import SoftProgram
    from tcn.types import Value
    p=arithmetic_scaffold((('x',F),),{'out':1},hidden=2);m=SoftProgram(p)
    out,_=m({'x':torch.tensor([.4])});out['out'].sum().backward()
    assert all(v.grad is not None for v in m.constants.values())
    frozen=m.export();assert frozen.is_frozen;assert not frozen.trainable_constants
    frozen.run({'x':Value.of(F,.4)})
