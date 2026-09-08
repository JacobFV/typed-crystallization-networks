import math
import pytest
import torch
from tcn.types import *
from tcn.operators import Registry
from tcn.learning import relaxed,tensor,ste
r=Registry();F=floating();I=integer(8)

def test_carriers_and_numeric_encodings():
    for t,x in [(BOOL,True),(integer(8),-128),(fixed(12,16),1.3125),(F,.5),(product(I,BOOL),(-3,False)),(setof(I,4),[-1,2,2])]:
        v=Value.of(t,x);assert Value.from_dict(v.to_dict())==v;assert Value.unflat(t,v.flat())==v
    with pytest.raises(OverflowError):Value.of(I,128)
    with pytest.raises(TypeError):Value.of(BOOL,1)
    with pytest.raises(ValueError):Type('object')
    assert Value.of(integer(8,overflow='wrap'),128).decoded==-128
    assert Value.of(integer(8,overflow='saturate'),128).decoded==127

def test_set_permutation_and_bounds():
    s=setof(integer(2,signed=False),3)
    assert Value.of(s,[0,1,2])==Value.of(s,[2,1,0,1])
    with pytest.raises(OverflowError):Value.of(s,[0,1,2,3])

def test_signature_constraints():
    for name,ts,out in [('and',(F,F),None),('sin',(F,F),None),('add',(F,F),BOOL),('fft',(setof(I),),None),('add',(I,F),None)]:
        with pytest.raises(TypeError):r.resolve(name,ts,out)
    with pytest.raises(TypeError):r.resolve('log',(floating(unit='m'),))
    assert r.resolve('mul',(floating(unit='m'),)*2).output.unit=='(m)^2'

@pytest.mark.parametrize('gate',range(16))
def test_all_truth_tables_exact_and_relaxed(gate):
    op=r.resolve(f'truth_{gate}',(BOOL,BOOL))
    for a in [False,True]:
        for b in [False,True]:
            values=[Value.of(BOOL,a),Value.of(BOOL,b)];expected=bool((gate>>(2*int(a)+int(b)))&1)
            assert r.exact(op,values).decoded==expected
            assert relaxed(r,op,[tensor(x) for x in values]).item()==float(expected)

@pytest.mark.parametrize('name,a,b,expected',[('add',5,2,7),('sub',5,2,3),('mul',5,2,10),('idiv',-5,2,-3),('mod',-5,2,1),('shl',3,2,12),('shr',-8,2,-2),('min',5,2,2),('max',5,2,5)])
def test_integer_operators(name,a,b,expected):
    assert r.exact(r.resolve(name,(I,I)),[Value.of(I,a),Value.of(I,b)]).decoded==expected

@pytest.mark.parametrize('name,x',[('sin',.3),('cos',.3),('exp',.3),('log',1.3),('sqrt',1.3),('abs',-.3),('neg',.3)])
def test_analytic_gradients(name,x):
    op=r.resolve(name,(F,));a=torch.tensor([x],requires_grad=True);y=relaxed(r,op,[a]);y.sum().backward()
    eps=1e-3;f=lambda v:r.exact(op,[Value.of(F,v)]).decoded
    assert a.grad.item()==pytest.approx((f(x+eps)-f(x-eps))/(2*eps),abs=2e-3)
    assert y.item()==pytest.approx(f(x),abs=1e-6)

def test_domain_errors_are_not_silent_clamps():
    with pytest.raises(ValueError):r.exact(r.resolve('log',(F,)),[Value.of(F,-1)])
    with pytest.raises(ValueError):relaxed(r,r.resolve('log',(F,)),[torch.tensor([-1.])])
    with pytest.raises(ValueError):r.exact(r.resolve('div',(F,F)),[Value.of(F,1),Value.of(F,0)])

def test_spectral_roundtrip():
    t=product(*(F for _ in range(8)));v=Value.of(t,tuple(math.sin(i) for i in range(8)))
    op=r.resolve('fft',(t,));z=r.exact(op,[v]);inv=r.exact(r.resolve('ifft',(z.type,)),[z])
    for expected,(real,imag) in zip(v.decoded,inv.decoded):assert real==pytest.approx(expected,abs=1e-6);assert imag==pytest.approx(0,abs=1e-6)
    assert torch.allclose(relaxed(r,op,[tensor(v)]),tensor(z),atol=1e-6)

def test_structural_temporal_and_conversion():
    s=setof(I,8);v=Value.of(s,[1,2]);two=Value.of(I,2)
    assert r.exact(r.resolve('member',(s,I)),[v,two]).decoded
    assert r.exact(r.resolve('remove',(s,I)),[v,two]).decoded==frozenset({1})
    pair=r.exact(r.resolve('pair',(s,s)),[v,v]);assert len(pair.decoded)==4
    t=product(I,I);p=Value.of(t,(3,5))
    assert r.exact(r.resolve('project',(t,),parameters={'index':1}),[p]).decoded==5
    assert r.exact(r.resolve('delay',(I,I)),[Value.of(I,3),Value.of(I,2)]).decoded==(2,3)
    bits=product(BOOL,BOOL,BOOL);packed=integer(3,signed=False)
    p=r.exact(r.resolve('pack',(bits,),packed),[Value.of(bits,(True,False,True))])
    assert p.raw==5
    assert r.exact(r.resolve('unpack',(packed,),bits),[p]).decoded==(True,False,True)

def test_ste_is_hard_forward_soft_backward():
    soft=torch.tensor([.3],requires_grad=True);y=ste(soft,torch.tensor([1.]));assert y.item()==1.;y.backward();assert soft.grad.item()==1.
