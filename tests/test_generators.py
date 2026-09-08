import copy
import pytest
from tcn.generation import *
from tcn.types import product,setof

@pytest.mark.parametrize('name',['logic','arithmetic','relations','signal','language','raster_text','geometry','world_2d','world_3d','computer','embodied_world'])
def test_common_contract_replay_restore_and_visibility(name,tmp_path):
    h=Host.create(name,seed=7,configuration={'resolution':16,'horizon':3});h.step(dt=.03)
    assert h.replay().digest==h.digest
    snap=h.snapshot();restored=Host.restore(snap)
    assert restored.digest==h.digest
    h.step(dt=.03);restored.step(dt=.03);assert h.digest==restored.digest
    path=tmp_path/(name+'.json.gz');h.save(path);assert Host.load(path).digest==h.digest
    v=h.view();assert not hasattr(v,'probes');assert not hasattr(v,'latent_states');assert not hasattr(v,'metadata')
    before=copy.deepcopy(h.state);v.objective['modified']=True;assert h.state==before

def test_ownership_and_availability():
    h=Host.create('logic');r=h.records[-1]
    r.observations={'agent_0/a':TimedValue(Value.of(BOOL,True),0,0),'agent_1/b':TimedValue(Value.of(BOOL,True),0,0),'public/future':TimedValue(Value.of(BOOL,True),2,2)}
    assert set(r.actor_view().observations)=={'a'}

def test_action_schema_and_simultaneous_conflicts():
    h=Host.create('logic')
    with pytest.raises(TypeError):h.step((Action('answer',arguments=(('value',Value.of(integer(),1)),)),))
    with pytest.raises(ValueError):h.step((Action('wait'),Action('wait')))
    with pytest.raises(ValueError):h.step(dt=0)

def test_composition_has_no_domain_branch_and_preserves_private_channels():
    cfg={'children':{'a':{'generator':'logic'},'b':{'generator':'arithmetic'}},'horizon':3}
    h=Host.create('composition',configuration=cfg);assert 'a.bits' in h.view().observations
    h.step((Action('a::answer',arguments=(('value',Value.of(BOOL,False)),)),));h.replay()
    assert Host.restore(h.snapshot()).generator.action_schema==h.generator.action_schema

def test_typed_composition_link_executes_a_transition():
    cfg={'children':{'a':{'generator':'logic'},'b':{'generator':'logic'}},'links':[{'source':'a','channel':'probes','port':'target','target':'b','verb':'answer','argument':'value'}]}
    h=Host.create('composition',configuration=cfg);record=h.step()
    child=Host.restore(h.state['children']['b']);assert child.records[-1].actions[0].verb=='answer'
    h.replay()

def test_computer_quoted_shell_write_read_and_keyboard():
    from generators.computer.generator import TEXT,PATH,KEY
    h=Host.create('computer',configuration={'horizon':6})
    command="echo 'alpha; beta > gamma' > /home/agent/result.txt"
    h.step((Action('command',arguments=(('text',text_value(command,512)),)),))
    h.step((Action('read',arguments=(('path',text_value('/home/agent/result.txt',128)),)),))
    assert h.state['result']['output']['content'].strip()=='alpha; beta > gamma'
    h.step((Action('type',arguments=(('text',text_value('cat /home/agent/result.txt',512)),)),))
    h.step((Action('key',arguments=(('code',Value.of(KEY,13)),)),))
    assert 'alpha; beta > gamma' in h.state['result']['output']['stdout']
    h.replay()

def test_physics_changes_state_and_communication_is_local():
    from generators.world_3d.generator import VEC3,TEXT
    h=Host.create('world_3d',configuration={'agents':2,'resolution':12,'horizon':8})
    before=h.state['poses']['agent_0']['position'][:]
    h.step((Action('move',arguments=(('force',Value.of(VEC3,(20.,0.,0.))),)),),dt=.1)
    assert h.state['poses']['agent_0']['position']!=before
    h.step((Action('say',arguments=(('text',text_value('hello',512)),)),),dt=.02)
    assert read_text(h.view('agent_1').observations['messages'])=='hello'
    assert read_text(h.view('agent_0').observations['messages'])==''
    h.step((Action('grasp',target='missing'),),dt=.02)
    assert read_text(h.view().observations['result'])=='out_of_reach'
    h.replay()

def test_paper_and_computer_share_embodied_action_contract():
    from generators.world_3d.generator import TEXT
    h=Host.create('embodied_world',configuration={'resolution':12,'horizon':8})
    h.step((Action('write',target='paper',arguments=(('text',text_value('42',512)),)),),dt=.01)
    assert next(o for o in h.state['objects'] if o['id']=='paper')['text']=='42'
    h.step((Action('read',target='paper'),),dt=.01);assert 'focus_pixels' in h.view().observations
    h.step((Action('type',target='computer',arguments=(('text',text_value('echo 42 > /home/agent/answer.txt',512)),)),),dt=.01)
    h.step((Action('key',target='computer',arguments=(('code',Value.of(integer(16,signed=False),13)),)),),dt=.01)
    assert h.state['computer']['result']['output']['exitCode']==0
    h.replay()
