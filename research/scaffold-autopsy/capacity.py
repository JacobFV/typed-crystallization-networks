"""Capacity test: fit the SAME arithmetic scaffold's prediction head on the task
targets by plain full-batch supervision, with no RL, no value head, no sampling.

Targets are exactly the trainer's: probes['gate'] = bits[0] xor bits[1] and
probes['target'] = gate xor invert. Inputs are the 8 scaffold features
(4 bits, goal, 2 action one-hot, dt) enumerated over all 16 relevant settings.
"""
import itertools
import json
import sys

import torch

from arith_joint import program
import tcn.learning as L
from tcn.learning import SoftProgram

# Test-local: let `tuple` concatenate operands with different batch ranks so the
# whole scaffold can be evaluated on a batch at once. Trainable constants are
# rank-1; batched inputs are rank-2. Semantics are unchanged for rank-1 inputs.
_relaxed = L.relaxed


def _batched_relaxed(registry, op, xs, temperature=1.):
    if op.name == 'tuple' and xs:
        rank = max(x.dim() for x in xs)
        shape = torch.broadcast_shapes(*(x.shape[:-1] for x in xs if x.dim() == rank))
        xs = [x.expand(*shape, x.shape[-1]) for x in xs]
    return _relaxed(registry, op, xs, temperature)


L.relaxed = _batched_relaxed


def dataset():
    rows = []
    for b0, b1, b2, b3, goal in itertools.product((0., 1.), repeat=5):
        gate = float(bool(b0) != bool(b1))
        target = float(bool(gate) != bool(goal))
        rows.append(({'bits': [b0, b1, b2, b3], 'goal': [goal], 'action': [1., 0.], 'dt': [1.]},
                     [target, gate]))
    return rows


def fit(steps=2000, lr=.04, hidden=8, seed=0, optimizer='adam'):
    m = SoftProgram(program(hidden=hidden, seed=seed))
    rows = dataset()
    x = {k: torch.tensor([[float(v) for v in r[0][k]] for r in rows]) for k in ('bits', 'goal', 'action', 'dt')}
    y = torch.tensor([r[1] for r in rows])
    params = [p for p in m.parameters() if p.requires_grad]
    opt = (torch.optim.Adam(params, lr=lr) if optimizer == 'adam' else torch.optim.SGD(params, lr=lr))
    torch.manual_seed(seed)
    trace = []
    for i in range(steps):
        out, _ = m(x)
        loss = (out['prediction'] - y).square().mean()
        opt.zero_grad(); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1e9))
        opt.step()
        if i % max(1, steps // 20) == 0 or i == steps - 1:
            trace.append({'step': i, 'loss': float(loss), 'grad': gn})
    out, _ = m(x)
    acc = float(((out['prediction'] > .5).float() == y).float().mean())
    return {'final_loss': float((out['prediction'] - y).square().mean()), 'accuracy': acc, 'trace': trace}


if __name__ == '__main__':
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    results = {}
    for hidden in (8, 32):
        for lr in (.04, .01):
            for seed in (0, 1, 2):
                k = f'hidden{hidden}_lr{lr}_seed{seed}'
                results[k] = fit(steps=steps, lr=lr, hidden=hidden, seed=seed)
                print(k, 'loss=%.5f acc=%.3f' % (results[k]['final_loss'], results[k]['accuracy']))
    with open('capacity.json', 'w') as f:
        json.dump(results, f, indent=2)
