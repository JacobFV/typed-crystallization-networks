from dataclasses import replace
import json
import subprocess
import pytest
import torch
from tcn.types import *
from tcn.graph import *
from tcn.operators import Registry
from tcn.learning import SoftProgram
from tcn.runtime import save_program,load_program,export_executable
from tcn.crystallize import Crystallizer

r=Registry()
def xor_program():
    return Program((('a',BOOL),('b',BOOL)),(Node('z',BOOL,tuple(Candidate(r.resolve(name,(BOOL,BOOL)),('a','b')) for name in ['and','or','xor'])),),(('out','z'),))

def test_graph_rejects_cycles_types_and_bad_depth():
    p=xor_program();n=p.nodes[0]
    with pytest.raises(ValueError):replace(p,nodes=(replace(n,candidates=(Candidate(n.candidates[0].operator,('z','b')),)),)).validate()
    with pytest.raises(TypeError):replace(p,inputs=(('a',integer()),('b',BOOL))).validate()
    with pytest.raises(ValueError):replace(p,input_depths=(('a',2),)).validate()

def test_recurrence_and_standalone_export(tmp_path):
    i=integer(16);op=r.resolve('add',(i,i))
    p=Program((('x',i),),(Node('next',i,(Candidate(op,('x','memory')),),selected=0),),(('sum','next'),),state=(('memory',Value.of(i,0),'next'),)).validate()
    st=None
    for x,want in [(1,1),(2,3),(3,6)]:out,st=p.run({'x':Value.of(i,x)},st);assert out['sum'].decoded==want
    path=export_executable(p,tmp_path/'agent.pyz')
    rows='\n'.join(json.dumps({'inputs':{'x':Value.of(i,x).to_dict()}}) for x in [1,2,3])+'\n'
    result=subprocess.run(['python3','-I',str(path)],input=rows,text=True,capture_output=True,check=True)
    assert [Value.from_dict(json.loads(row)['outputs']['sum']).decoded for row in result.stdout.splitlines()]==[1,3,6]

def test_frozen_module_reuse_and_integrity(tmp_path):
    p=xor_program().harden({'z':2});name=r.register_module(p)
    op=r.resolve(name,(BOOL,BOOL));q=Program(p.inputs,(Node('call',op.output,(Candidate(op,('a','b')),),selected=0),),(('result','call'),)).validate(r)
    # A single-output module carries that output's type directly; only a module
    # with several outputs forms a tuple. The former one-field product cost every
    # call site a `project` node.
    assert op.output==BOOL
    assert q.run({'a':Value.of(BOOL,True),'b':Value.of(BOOL,False)},registry=r)[0]['result'].decoded is True
    model=SoftProgram(q,r);a=torch.tensor([1.],requires_grad=True);out,_=model({'a':a,'b':torch.tensor([0.])});assert not out['result'].requires_grad
    path=save_program(q,tmp_path/'module.json',r);restored,lib=load_program(path);assert restored==q
    assert q.description_bits(r)>q.description_bits()
    d=json.loads(path.read_text());d['digest']='bad';path.write_text(json.dumps(d))
    with pytest.raises(ValueError):load_program(path)

def test_learning_and_progressive_crystallization():
    from examples.mixed import problem
    from tcn.synthesis import fit
    p,signals,examples=problem();model,report=fit(p,examples,signals,steps=250,tolerance=.005)
    assert report['exact_conformance'];assert report['fully_frozen']
    assert any(e['accepted'] for e in report['freeze_events'])
    # Untouched numeric values verify composition beyond the training grid.
    import math
    for x in [.13,-.43]:
        out,_=model.export().run({'a':Value.of(BOOL,True),'b':Value.of(BOOL,False),'x':Value.of(floating(),x)})
        assert out['answer'].decoded==pytest.approx(math.sin(1+x),abs=1e-6)

def test_freeze_rollback_preserves_optimizer_and_gradients():
    p=xor_program();m=SoftProgram(p);opt=torch.optim.Adam(m.parameters(),lr=.1)
    def loss():out,_=m({'a':torch.tensor([1.]),'b':torch.tensor([0.])});return (out['out']-.6).square().mean()
    scheduler=Crystallizer(m,opt,tolerance=0)
    before=m.choices[0].detach().clone();event=scheduler.try_freeze('z',loss,retrain_steps=0)
    assert not event.accepted;assert 'z' not in m.frozen;assert m.choices[0].requires_grad;assert torch.equal(m.choices[0],before)
