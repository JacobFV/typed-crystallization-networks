"""Log per-term gradient norms/conflicts for the arithmetic scaffold and for examples/joint.py."""
import json
import sys

import torch

from arith_joint import trainer as arith_trainer
from instrument import InstrumentedTrainer


def instrument(t, episodes):
    t.__class__ = InstrumentedTrainer
    torch.manual_seed(t.config.seed)
    out = []
    for i in range(episodes):
        out.append(t.instrumented_step(i))
    return out


def agg(rows, key, sub):
    return sum(r[key][sub] for r in rows) / len(rows)


def report(name, rows):
    print(f'== {name}  ({len(rows)} episodes)')
    print('  mean return          ', round(sum(r['return'] for r in rows) / len(rows), 4))
    for k in rows[0]['grad_norm']:
        print(f'  |grad {k:<11}|      ', round(agg(rows, 'grad_norm', k), 5))
    for k in rows[0]['cosine']:
        v = agg(rows, 'cosine', k)
        if abs(v) > 1e-9:
            print(f'  cos {k:<22}', round(v, 4))
    print('  mean pre-clip norm    ', round(sum(r['clipped_from'] for r in rows) / len(rows), 4))
    print('  frac clipped          ', round(sum(r['clipped_from'] > 5. for r in rows) / len(rows), 4))
    print('  mean |policy logit|max', round(sum(r['logit_absmax'] for r in rows) / len(rows), 4))
    print('  mean loss             ', {k: round(agg(rows, 'loss', k), 4) for k in rows[0]['loss']})


def main():
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    a = arith_trainer(episodes=episodes, seed=0)
    rows_a = instrument(a, episodes)
    report('arithmetic scaffold', rows_a)
    print('  by-parameter grad norms (episode mean):')
    heads = rows_a[0]['by_parameter'].keys()
    for h in heads:
        p = sum(r['by_parameter'][h]['prediction'] for r in rows_a) / len(rows_a)
        v = sum(r['by_parameter'][h]['value'] for r in rows_a) / len(rows_a)
        ac = sum(r['by_parameter'][h]['actor'] for r in rows_a) / len(rows_a)
        if max(p, v, ac) > 1e-6:
            print(f'    {h:<24} pred={p:.5f} value={v:.5f} actor={ac:.5f}')

    from examples.joint import trainer as joint_trainer
    b = joint_trainer(episodes=episodes, seed=0)
    rows_b = instrument(b, episodes)
    report('examples/joint.py typed logic scaffold', rows_b)

    with open(f'instrument_{episodes}.json', 'w') as f:
        json.dump({'arithmetic': rows_a, 'joint': rows_b}, f, default=str)


if __name__ == '__main__':
    main()
