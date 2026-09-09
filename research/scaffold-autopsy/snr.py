"""Gradient signal-to-noise ratio of each objective term at batch size one.

For a term L, with per-episode gradients g_i, SNR = ||mean(g_i)|| / mean(||g_i||).
A value near 1 means one episode already points where the average points; a value
near 0 means a single-episode update is almost entirely noise, and the useful
component only appears after averaging ~1/SNR^2 episodes.
"""
import json
import sys

import torch

from arith_joint import trainer as arith_trainer
from instrument import InstrumentedTrainer


def snr(t, episodes, warmup=0):
    t.__class__ = InstrumentedTrainer
    torch.manual_seed(t.config.seed)
    for i in range(warmup):
        t.instrumented_step(i)
    active = [p for p in t.model.parameters() if p.requires_grad]
    acc = {}
    for i in range(warmup, warmup + episodes):
        terms, _, _, _ = t.terms(i)
        for k, v in terms.items():
            if not (isinstance(v, torch.Tensor) and v.requires_grad):
                continue
            g = torch.autograd.grad(v, active, retain_graph=True, allow_unused=True)
            flat = torch.cat([(torch.zeros_like(p) if x is None else x).flatten() for x, p in zip(g, active)])
            s = acc.setdefault(k, {'sum': torch.zeros_like(flat), 'norms': 0., 'n': 0})
            s['sum'] += flat
            s['norms'] += float(flat.norm())
            s['n'] += 1
    out = {}
    for k, s in acc.items():
        mean_norm = float((s['sum'] / s['n']).norm())
        norm_mean = s['norms'] / s['n']
        out[k] = {'||mean g||': round(mean_norm, 5), 'mean ||g||': round(norm_mean, 5),
                  'snr': round(mean_norm / norm_mean, 5) if norm_mean > 0 else 0.,
                  'episodes_to_signal': round((norm_mean / mean_norm) ** 2, 1) if mean_norm > 0 else None}
    return out


def main():
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    results = {}
    for warmup in (0, 200):
        t = arith_trainer(episodes=episodes + warmup + 1, seed=0)
        results[f'arithmetic_warmup{warmup}'] = snr(t, episodes, warmup)
        print(f'arithmetic scaffold, after {warmup} training episodes:')
        for k, v in results[f'arithmetic_warmup{warmup}'].items():
            print(f'  {k:<11} SNR={v["snr"]:.4f}  ||mean g||={v["||mean g||"]:.4f}  '
                  f'mean||g||={v["mean ||g||"]:.4f}  episodes/update for SNR~1: {v["episodes_to_signal"]}',
                  flush=True)
    from examples.joint import trainer as joint_trainer
    for warmup in (0, 40):
        t = joint_trainer(episodes=episodes + warmup + 1, seed=0)
        results[f'joint_warmup{warmup}'] = snr(t, episodes, warmup)
        print(f'examples/joint.py typed logic scaffold, after {warmup} training episodes:')
        for k, v in results[f'joint_warmup{warmup}'].items():
            print(f'  {k:<11} SNR={v["snr"]:.4f}  ||mean g||={v["||mean g||"]:.4f}  '
                  f'mean||g||={v["mean ||g||"]:.4f}  episodes/update for SNR~1: {v["episodes_to_signal"]}',
                  flush=True)
    with open('snr.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
