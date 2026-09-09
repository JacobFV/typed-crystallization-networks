"""Ablations over the arithmetic-scaffold joint run.

Each entry is (name, kwargs for arith_joint.trainer). Everything is run at the
same episode budget and the same seeds so the numbers are comparable.
"""
import json
import sys
import time

import torch

from arith_joint import trainer, evaluate, summarize

ABLATIONS = {
    'baseline':          dict(),
    'value_weight=0':    dict(value_weight=0.),
    'lr=0.01':           dict(lr=.01),
    'lr=0.005':          dict(lr=.005),
    'hidden=32':         dict(hidden=32),
    'probe_head':        dict(probe=True),
    'entropy=0.05':      dict(entropy_weight=.05),
    'discount=0':        dict(discount=0.),
    'policy_weight=4':   dict(policy_weight=4.),
    # isolate the supervised half: policy and value practically switched off
    'pred_only':         dict(policy_weight=1e-8, value_weight=0., entropy_weight=0.),
    'pred_only_lr.01':   dict(policy_weight=1e-8, value_weight=0., entropy_weight=0., lr=.01),
    'no_value_head':     dict(value_weight=0., entropy_weight=.01),
}


def run(name, kwargs, episodes, seeds):
    out = []
    for seed in seeds:
        torch.manual_seed(seed)
        t = trainer(episodes=episodes, seed=seed, **kwargs)
        h = t.run()
        s = summarize(h)
        s['deterministic_eval_return'] = evaluate(t)
        s['seed'] = seed
        out.append(s)
    mean = lambda k: sum(o[k] for o in out) / len(out)
    return {'runs': out,
            'mean_final_prediction_loss': mean('final_prediction_loss'),
            'mean_final_return': mean('final_mean_return'),
            'mean_eval_return': mean('deterministic_eval_return')}


def main():
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 320
    seeds = [int(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['0', '1'])]
    names = sys.argv[3].split(',') if len(sys.argv) > 3 else list(ABLATIONS)
    results = {}
    for name in names:
        t0 = time.time()
        results[name] = run(name, ABLATIONS[name], episodes, seeds)
        r = results[name]
        print(f"{name:<20} pred={r['mean_final_prediction_loss']:.4f} "
              f"train_return={r['mean_final_return']:.3f} eval_return={r['mean_eval_return']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    with open(f'ablate_{episodes}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == '__main__':
    main()
