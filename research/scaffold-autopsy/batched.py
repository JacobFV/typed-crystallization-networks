"""Test the minibatch hypothesis without touching the core yet.

JointTrainer.run() takes exactly one optimizer step per episode. This wrapper
accumulates the identical per-episode loss over `batch` consecutive episodes and
takes one step, leaving every loss term, weight, and learning rate unchanged.
The episode budget (number of environment episodes) is held fixed, so a larger
batch means proportionally FEWER optimizer steps -- a strictly harder setting for
the batched runs if the problem were step count rather than gradient noise.
"""
import json
import sys
import time

import torch

from arith_joint import trainer, evaluate
from tcn.training import JointTrainer


def run_batched(t, batch):
    torch.manual_seed(t.config.seed)
    i = 0
    steps = 0
    while i < t.config.episodes:
        n = min(batch, t.config.episodes - i)
        t.optimizer.zero_grad()
        total = None
        for _ in range(n):
            loss = t.episode(i, train=True, loss_only=True)
            total = loss if total is None else total + loss
            i += 1
        (total / n).backward()
        torch.nn.utils.clip_grad_norm_(t.model.parameters(), 5.)
        t.optimizer.step()
        steps += 1
    t.completed = t.config.episodes
    return steps


def held_out(t, episodes=16):
    losses = []
    for i in range(episodes):
        m, _ = t.episode(i, train=False, split='test')
        losses.append(m['prediction_loss'])
    return sum(losses) / len(losses)


def main():
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 640
    batches = [int(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['1', '8', '32'])]
    seeds = [int(x) for x in (sys.argv[3].split(',') if len(sys.argv) > 3 else ['0', '1'])]
    kw = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    out = {}
    for b in batches:
        rets, preds = [], []
        t0 = time.time()
        for s in seeds:
            t = trainer(episodes=episodes, seed=s, **kw)
            steps = run_batched(t, b)
            rets.append(evaluate(t, episodes=16))
            preds.append(held_out(t))
        out[f'batch={b}'] = {'eval_returns': rets, 'mean_return': sum(rets) / len(rets),
                             'held_out_prediction_loss': sum(preds) / len(preds),
                             'optimizer_steps': steps}
        r = out[f'batch={b}']
        print(f"batch={b:<4} steps={steps:<5} eval_return={r['mean_return']:.3f} "
              f"pred_loss={r['held_out_prediction_loss']:.5f} returns={rets} ({time.time() - t0:.0f}s)",
              flush=True)
    tag = '_'.join(f'{k}{v}' for k, v in kw.items()) or 'default'
    with open(f'batched_{episodes}_{tag}.json', 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
