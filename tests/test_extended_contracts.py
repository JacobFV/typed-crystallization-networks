import pytest
import torch
from tcn.types import *
from tcn.operators import Registry
from tcn.graph import Program,Node,Candidate
from tcn.learning import SoftProgram,relaxed,tensor
r=Registry();F=floating();I=integer(8)

@pytest.mark.parametrize('name,a,b,expected',[('div',4.,2.,2.),('pow',4.,.5,2.),('atan2',1.,1.,.7853981634)])
def test_binary_analytic(name,a,b,expected):
    op=r.resolve(name,(F,F));xs=[Value.of(F,a),Value.of(F,b)]
    assert r.exact(op,xs).decoded==pytest.approx(expected,abs=1e-6)
    assert relaxed(r,op,[tensor(x) for x in xs]).item()==pytest.approx(expected,abs=1e-6)

@pytest.mark.parametrize('name,expected',[('sum',6.),('mean',2.),('reduce_min',1.),('reduce_max',3.),('count',3)])
def test_reductions(name,expected):
    t=product(F,F,F);v=Value.of(t,(1.,2.,3.));op=r.resolve(name,(t,))
    assert r.exact(op,[v]).decoded==expected
    assert relaxed(r,op,[tensor(v)]).item()==expected

def test_sparse_general_sets_and_categorical_lifts():
    t=setof(product(integer(64,signed=False,role='category'),F),4)
    values=[(2**55,.25),(3,1.75)]
    v=Value.of(t,values);assert Value.unflat(t,v.flat())==v
    assert v.flat()==Value.of(t,list(reversed(values))).flat()
    assert t.width==4*(1+64+1)

def test_set_operators_and_frozen_map_filter():
    t=setof(I,4);v=Value.of(t,[1,2]);other=Value.of(t,[2,3])
    assert r.exact(r.resolve('union',(t,t)),[v,other]).decoded=={1,2,3}
    assert r.exact(r.resolve('intersection',(t,t)),[v,other]).decoded=={2}
    assert r.exact(r.resolve('insert',(t,I)),[v,Value.of(I,4)]).decoded=={1,2,4}
    neg=Program((('x',I),),(Node('y',I,(Candidate(r.resolve('neg',(I,)),('x',)),),selected=0),),(('out','y'),))
    name=r.register_module(neg)
    assert r.exact(r.resolve('map',(t,),parameters={'module':name}),[v]).decoded=={-1,-2}
    pred=Program((('x',I),),(Node('y',BOOL,(Candidate(r.resolve('gt',(I,I)),('x','zero')),),selected=0),),(('out','y'),),constants=(('zero',Value.of(I,0)),))
    name=r.register_module(pred)
    assert r.exact(r.resolve('filter',(t,),parameters={'module':name}),[Value.of(t,[-1,2])]).decoded=={2}
    edge=product(I,I);s=setof(edge,4)
    op=r.resolve('join',(s,s),parameters={'left':1,'right':0})
    result=r.exact(op,[Value.of(s,[(1,2)]),Value.of(s,[(2,3),(4,5)])])
    assert result.decoded=={((1,2),(2,3))}

def test_recurrent_module_lifts_state_and_remains_callable():
    p=Program((('x',I),),(Node('next',I,(Candidate(r.resolve('add',(I,I)),('x','state')),),selected=0),),(('out','next'),),state=(('state',Value.of(I,0),'next'),))
    name=r.register_module(p);m=r.modules[name];assert not m.state
    result=r.exact(r.resolve(name,(I,I)),[Value.of(I,3),Value.of(I,4)])
    assert result.decoded==(7,7)

def test_batched_frozen_constant_first_and_precision_contract():
    p=Program((('x',F),),(Node('out',F,(Candidate(r.resolve('add',(F,F)),('bias','x')),),selected=0),),(('out','out'),),constants=(('bias',Value.of(F,1.)),))
    m=SoftProgram(p);out,_=m({'x':torch.tensor([[1.],[2.],[3.]])});assert out['out'].tolist()==[[2.],[3.],[4.]]
    p=Program((('x',F),),(Node('out',F,(Candidate(r.resolve('identity',(F,)),('x',)),)),),(('out','out'),))
    m=SoftProgram(p);m.quantization['out']=(fixed(),1.)
    with pytest.raises(ValueError):m({'x':torch.tensor([.2])})

def test_typed_action_sampling_supports_continuous_discrete_and_sets():
    import random
    from tcn.policy import sample_typed,sample_exact,parameter_width
    t=product(BOOL,integer(3,signed=False,role='category'),floating(bounds=(-2.,2.)),setof(BOOL,2))
    params=torch.randn(parameter_width(t),requires_grad=True)
    value,logp,entropy=sample_typed(t,params);assert value.type==t
    logp.backward();assert params.grad is not None;assert torch.isfinite(params.grad).all()
    exact=sample_exact(t,params.detach().tolist(),random.Random(0),deterministic=True)
    soft,_,_=sample_typed(t,params,deterministic=True)
    assert exact.raw[0:2]==soft.raw[0:2];assert exact.decoded[2]==pytest.approx(soft.decoded[2],abs=1e-6)
