"""Decisive test: the SAME arithmetic scaffold under the patched trainer.

usage: fixed.py EPISODES BATCH SEEDS [extra-json]

Everything except `batch` (episodes averaged per optimizer step) matches the
recorded failing configuration: same scaffold, same generator settings, same
objectives, same lr, same loss weights, same horizon.
"""
import json
import sys
import time

from arith_joint import trainer, evaluate, summarize
from batched import held_out


def main():
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 5120
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seeds = [int(x) for x in (sys.argv[3].split(',') if len(sys.argv) > 3 else ['0'])]
    extra = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    out = []
    for s in seeds:
        t0 = time.time()
        t = trainer(episodes=episodes, seed=s, batch=batch, **extra)
        h = t.run()
        r = summarize(h, window=64)
        r['deterministic_eval_return'] = evaluate(t, episodes=16)
        r['held_out_prediction_loss'] = held_out(t, episodes=16)
        r['optimizer_steps'] = (episodes + batch - 1) // batch
        r['batch'] = batch
        r['seed'] = s
        r['seconds'] = round(time.time() - t0, 1)
        out.append(r)
        print(json.dumps(r), flush=True)
        with open(f'fixed_ep{episodes}_b{batch}_s{s}.json', 'w') as f:
            json.dump({'summary': r, 'history': h}, f, default=str)
    m = lambda k: sum(o[k] for o in out) / len(out)
    print(f'SUMMARY batch={batch} episodes={episodes} '
          f'eval_return={m("deterministic_eval_return"):.3f} '
          f'held_out_pred={m("held_out_prediction_loss"):.5f} '
          f'train_return={m("final_mean_return"):.3f}', flush=True)


if __name__ == '__main__':
    main()
