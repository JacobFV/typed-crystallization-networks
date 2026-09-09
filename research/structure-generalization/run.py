"""Run one structure-generalization condition over several seeds.

    .venv/bin/python research/structure-generalization/run.py <condition> [seeds]

Writes research/structure-generalization/out/<condition>.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import torch
from tcn.generation import Host
import common as C

EVAL = range(10000, 10064)          # held-out episode addresses, split='test'
EVAL16 = range(10000, 10016)        # the record's evaluation budget


def build(kind, schedule, seed, episodes, tau=None):
    settings = schedule(0, 'train')
    host = Host.create('logic', configuration=settings)
    if kind == 'record':
        names, model = C.record_scaffold(host)
    elif kind == 'lookup':
        names, model = C.program_scaffold(host, interpreter_only=False)
    elif kind == 'interpreter':
        names, model = C.program_scaffold(host, interpreter_only=True)
    elif kind == 'interpreter_wired':
        names, model = C.program_scaffold(host, interpreter_only=True, wired=True)
    elif kind == 'lookup_wired':
        names, model = C.program_scaffold(host, interpreter_only=False, wired=True)
    else:
        raise ValueError(kind)
    C.residual_init(model)
    if tau is not None:
        model.temperatures['relation'] = tau
    config = C.make_config(names, settings, episodes, seed)
    return C.ScheduledTrainer(model, config, schedule)


# condition -> (scaffold kind, train schedule, episodes, {eval name: eval schedule})
def conditions():
    fixed_xor = C.fixed_schedule({'depth': 1, 'table': 6, 'fixed_inputs': True})
    affine = C.table_schedule(C.AFFINE)
    nonaffine = C.table_schedule(C.NONAFFINE)
    affine_w = C.table_schedule(C.AFFINE, base={'depth': 1, 'fixed_inputs': False})
    nonaffine_w = C.table_schedule(C.NONAFFINE, base={'depth': 1, 'fixed_inputs': False})
    d12 = C.depth_schedule([1, 2])
    depth = {f'depth_{d}': C.depth_schedule([d]) for d in (1, 2, 3, 4)}
    table_evals = {'train_tables_affine': affine, 'heldout_tables_nonaffine': nonaffine}
    table_evals_r = {'train_tables_nonaffine': nonaffine, 'heldout_tables_affine': affine}
    wired_evals = {'train_tables_affine_wired': affine_w, 'heldout_tables_nonaffine_wired': nonaffine_w}
    return {
        # 1. reproduction of the validation record
        'record_fixed_xor': ('record', fixed_xor, 160, {'fixed_xor': fixed_xor}),
        # 2. same scaffold, many gate families
        'record_multitable_affine': ('record', affine, 320, table_evals),
        'record_multitable_nonaffine': ('record', nonaffine, 320, table_evals_r),
        # 3. table-conditioned scaffolds
        'lookup_affine': ('lookup', affine, 320, table_evals),
        'lookup_nonaffine': ('lookup', nonaffine, 320, table_evals_r),
        'interpreter_affine': ('interpreter', affine, 320, table_evals),
        'interpreter_nonaffine': ('interpreter', nonaffine, 320, table_evals_r),
        # 3b. also generalizing over gate wiring
        'lookup_wired_affine': ('lookup_wired', affine_w, 320, wired_evals),
        'interpreter_wired_affine': ('interpreter_wired', affine_w, 320, wired_evals),
        # 4. depth generalization, record scaffold, random wiring and tables
        'record_depth12': ('record', d12, 320, depth),
    }


def main():
    name = sys.argv[1]
    seeds = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else list(range(8))
    kind, schedule, episodes, evals = conditions()[name]
    torch.set_num_threads(1)
    out = {'condition': name, 'scaffold': kind, 'episodes': episodes, 'seeds': seeds,
           'eval_indices': [EVAL.start, EVAL.stop], 'runs': [], 'baselines': {}}
    for ename, esched in evals.items():
        out['baselines'][ename] = C.baselines(EVAL, 'test', esched)
    if name == 'record_fixed_xor':
        out['baselines']['fixed_xor_16'] = C.baselines(EVAL16, 'test', evals['fixed_xor'])
    t0 = time.time()
    for seed in seeds:
        trainer = build(kind, schedule, seed, episodes)
        history = trainer.run()
        row = {'seed': seed,
               'train_return_last32': sum(h['return'] for h in history[-32:]) / 32,
               'final_prediction_loss': sum(h['prediction_loss'] for h in history[-8:]) / 8,
               'initial_prediction_loss': sum(h['prediction_loss'] for h in history[:8]) / 8,
               'selections': trainer.model.selections(), 'evaluations': {}}
        for ename, esched in evals.items():
            r = C.deterministic_returns(trainer, EVAL, 'test', esched)
            row['evaluations'][ename] = {'mean': sum(r) / len(r), 'returns': r}
        if name == 'record_fixed_xor':
            r = C.deterministic_returns(trainer, EVAL16, 'test', evals['fixed_xor'])
            row['evaluations']['fixed_xor_16'] = {'mean': sum(r) / len(r), 'returns': r}
        out['runs'].append(row)
        print(f"{name} seed {seed}: " + "  ".join(f"{k}={v['mean']:.3f}" for k, v in row['evaluations'].items()), flush=True)
    out['seconds'] = time.time() - t0
    out['summary'] = {k: C.summarize([r['evaluations'][k]['mean'] for r in out['runs']])
                      for k in out['runs'][0]['evaluations']}
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / f'{name}.json').write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out['summary'], indent=2))


if __name__ == '__main__':
    main()
