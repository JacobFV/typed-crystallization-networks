"""Reproduce the recorded near-chance arithmetic-scaffold result."""
import json
import sys
import time

from arith_joint import trainer, evaluate, summarize


def main():
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 640
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    t0 = time.time()
    t = trainer(episodes=episodes, seed=seed)
    print('parameters:', sum(p.numel() for p in t.model.parameters() if p.requires_grad),
          'nodes:', len(t.model.program.nodes))
    history = t.run()
    s = summarize(history)
    s['deterministic_eval_return'] = evaluate(t)
    s['seconds'] = round(time.time() - t0, 1)
    s['seed'] = seed
    print(json.dumps(s, indent=2))
    with open(f'repro_seed{seed}_ep{episodes}.json', 'w') as f:
        json.dump({'summary': s, 'history': history}, f, indent=2, default=str)


if __name__ == '__main__':
    main()
