"""Crystallize a trained scaffold and evaluate the exact frozen agent on held-out gate families.

    .venv/bin/python research/structure-generalization/crystal.py <record|lookup> [seeds]
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent))
import torch
from tcn.generation import Host
from tcn.crystallize import Crystallizer
from tcn.agent import Agent
import common as C
from run import build

EVAL = range(10000, 10064)


def frozen_returns(program, registry, config, indices, schedule, horizon=4):
    agent = Agent(program, registry, config)
    out = []
    for i in indices:
        goal = C.OBJECTIVES[i % len(C.OBJECTIVES)]
        host = Host.create('logic', seed=config.seed, index=i, split='test',
                           configuration=dict(schedule(i, 'test')) | {'horizon': horizon}, objective=goal)
        agent.rollout(host, deterministic=True)
        out.append(sum(sum(v.decoded for v in r.reward_components.values()) for r in host.records))
    return out


def main():
    kind = sys.argv[1]
    seeds = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else list(range(8))
    torch.set_num_threads(1)
    train = C.table_schedule(C.AFFINE)
    evals = {'train_tables_affine': train, 'heldout_tables_nonaffine': C.table_schedule(C.NONAFFINE)}
    result = {'scaffold': kind, 'seeds': seeds, 'runs': [],
              'baselines': {k: C.baselines(EVAL, 'test', v) for k, v in evals.items()}}
    for seed in seeds:
        t = build(kind, train, seed, 320)
        t.run()
        sched = Crystallizer(t.model, t.optimizer, tolerance=.05, entropy_limit=.9)
        sched.run(lambda: sum(t.episode(20000 + i, False, 'validation', loss_only=True) for i in range(2)) / 2,
                  rounds=24, retrain_steps=2)
        frozen = len(t.model.frozen) == len(t.model.program.nodes) and all(not p.requires_grad for p in t.model.constants.values())
        row = {'seed': seed, 'fully_frozen': bool(frozen),
               'frozen_nodes': len(t.model.frozen), 'total_nodes': len(t.model.program.nodes),
               'selections': t.model.selections(), 'evaluations': {}}
        if frozen:
            program = t.model.export()
            for name, s in evals.items():
                r = frozen_returns(program, t.model.registry, t.config, EVAL, s)
                row['evaluations'][name] = {'mean': sum(r) / len(r), 'returns': r}
        result['runs'].append(row)
        print(f"{kind} seed {seed} frozen={frozen} " +
              " ".join(f"{k}={v['mean']:.3f}" for k, v in row['evaluations'].items()), flush=True)
    done = [r for r in result['runs'] if r['fully_frozen']]
    if done:
        result['summary'] = {k: C.summarize([r['evaluations'][k]['mean'] for r in done]) for k in done[0]['evaluations']}
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / f'crystal_{kind}.json').write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result.get('summary', {}), indent=2))


if __name__ == '__main__':
    main()
