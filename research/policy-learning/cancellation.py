"""The uniform 16-way truth-table mixture is exactly derivative-blocking.

relaxed truth_k(a,b) = sum_i v_i * ((k>>i)&1) with v = ((1-a)(1-b),(1-a)b,a(1-b),ab).
Averaging over all 16 tables gives each v_i a coefficient of 1/2, so the mixture is
0.5 for every (a,b) and its Jacobian is identically zero. Verified numerically.
"""
import torch, json
from tcn.operators import Registry
from tcn.learning import relaxed
r=Registry()
out={}
for a0,b0 in ((0.,0.),(1.,0.),(0.3,0.7),(1.,1.)):
    a=torch.tensor([a0],requires_grad=True);b=torch.tensor([b0],requires_grad=True)
    ys=[relaxed(r,r.resolve(f'truth_{k}',(a.new_zeros(1).shape and __import__('tcn.types',fromlist=['BOOL']).BOOL,)*2),[a,b]) for k in range(16)]
    y=sum(ys)/16
    ga,gb=torch.autograd.grad(y.sum(),[a,b])
    out[f'{a0},{b0}']={'mixture_value':float(y),'d_da':float(ga),'d_db':float(gb)}
# and with one table given weight (softmax over logits with p[12]=2)
logits=torch.zeros(16);logits[12]=2.
w=torch.softmax(logits,0)
for a0,b0 in ((0.3,0.7),):
    a=torch.tensor([a0],requires_grad=True);b=torch.tensor([b0],requires_grad=True)
    from tcn.types import BOOL
    ys=[relaxed(r,r.resolve(f'truth_{k}',(BOOL,BOOL)),[a,b]) for k in range(16)]
    y=sum(wi*yi for wi,yi in zip(w,ys))
    ga,gb=torch.autograd.grad(y.sum(),[a,b])
    out['p12=2 @0.3,0.7']={'mixture_value':float(y),'d_da':float(ga),'d_db':float(gb)}
print(json.dumps(out,indent=1))
json.dump(out,open('research/policy-learning/out/cancellation.json','w'),indent=1)
