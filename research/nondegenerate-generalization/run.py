"""Step 4: structural generalization on a non-degenerate benchmark.

Track 4 established that the recorded scaffold cannot generalize over gate
families and that a table-conditioned candidate fixes it. Both were measured on
the benchmark track 3 later showed to be degenerate, and track 4's own "affine"
training pool `(0,3,5,6,9,10,12,15)` contains all six tables whose output
ignores an input -- so three quarters of that pool were constants or
projections, and a learner could score on them without reading the table at all.

This re-runs the experiment on the ten tables that genuinely depend on both
inputs, split into two disjoint held-out halves, and reports the discrete
enumeration baseline beside every gradient number.

    .venv/bin/python research/nondegenerate-generalization/run.py [seeds]
"""
from __future__ import annotations
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'structure-generalization'))
sys.path.insert(0, str(HERE.parent.parent))

import torch
from tcn.generation import Host
import common as C

# Two disjoint pools drawn from the ten tables whose output depends on both
# arguments. Each is *balanced*: across its members every input assignment is
# true exactly as often as false, so no fixed input-independent gate beats 0.500
# accuracy and the analytic ceiling is exactly 2.00/4. Verified by exhaustive
# search over all 16 gates and both objectives.
#
# Stated limitation: only two non-degenerate tables are affine (6 and 9), and no
# balanced disjoint split places one in each pool -- exhaustive search over all
# balanced 4-subsets returns none. Pool B therefore holds both. Both directions
# are run so that confound is symmetric and visible rather than one-sided.
POOL_A = (1, 2, 13, 14)
POOL_B = (4, 6, 9, 11)
EVAL = range(10000, 10064)
TRAIN_PROBE = range(20000, 20008)     # addresses used only by the enumerator

def schedules(wired):
    base = {'depth': 1, 'inputs': 2, 'fixed_inputs': not wired, 'nondegenerate': True}
    return (C.table_schedule(POOL_A, base=base), C.table_schedule(POOL_B, base=base))

def build(kind, schedule, seed, episodes):
    settings = schedule(0, 'train')
    host = Host.create('logic', configuration=settings)
    if kind == 'record':
        names, model = C.record_scaffold(host)
    elif kind == 'lookup':
        names, model = C.program_scaffold(host, interpreter_only=False)
    elif kind == 'lookup_wired':
        names, model = C.program_scaffold(host, interpreter_only=False, wired=True)
    else:
        raise ValueError(kind)
    C.residual_init(model)
    return C.ScheduledTrainer(model, C.make_config(names, settings, episodes, seed), schedule)

def enumerate_policy(trainer, train_schedule, test_schedule):
    """Best discrete program by training return, then its held-out return.

    Searches exactly the scaffold's own candidate space, scoring by actual
    return on training episodes -- the same signal the gradient run optimizes.
    """
    model = trainer.model
    choosing = [n for n in model.program.nodes if len(n.candidates) > 1]
    counts = [len(n.candidates) for n in choosing]
    started = time.perf_counter(); best = None; evaluated = 0
    for combination in itertools.product(*(range(c) for c in counts)):
        model.trials = {n.name: c for n, c in zip(choosing, combination)}
        evaluated += 1
        got = C.deterministic_returns(trainer, TRAIN_PROBE, 'train', train_schedule)
        mean = statistics.mean(got)
        if best is None or mean > best[0]:
            best = (mean, dict(model.trials))
    model.trials = dict(best[1])
    held = C.deterministic_returns(trainer, EVAL, 'test', test_schedule)
    model.trials = {}
    return {'space_size': int(torch.tensor(counts).prod()), 'evaluated': evaluated,
            'train_return': best[0], 'heldout_return': statistics.mean(held),
            'seconds': time.perf_counter() - started,
            'selections': {k: int(v) for k, v in best[1].items()}}

def run(kind, direction, seeds, episodes=320):
    a, b = schedules(kind.endswith('wired'))
    train_schedule, heldout_schedule = (a, b) if direction == 'ab' else (b, a)
    rows = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        trainer = build(kind, train_schedule, seed, episodes)
        trainer.run()
        seen = C.deterministic_returns(trainer, EVAL, 'test', train_schedule)
        unseen = C.deterministic_returns(trainer, EVAL, 'test', heldout_schedule)
        selections = trainer.model.selections()
        interpreter = None
        if kind.startswith('lookup'):
            node = next(n for n in trainer.model.program.nodes if n.name == 'relation')
            interpreter = selections['relation'] == len(node.candidates) - 1
        rows.append({'seed': seed, 'seen': statistics.mean(seen), 'unseen': statistics.mean(unseen),
                     'interpreter_selected': interpreter})
        print(f"  {kind:13s} {direction} seed {seed}: seen {rows[-1]['seen']:.2f} "
              f"unseen {rows[-1]['unseen']:.2f} interpreter={interpreter}", flush=True)
    return rows


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    out = {}
    for kind in ('record', 'lookup', 'lookup_wired'):
        for direction in ('ab', 'ba'):
            key = f'{kind}_{direction}'
            print(f"== {key} ==", flush=True)
            out[key] = {'rows': run(kind, direction, seeds)}
    # every return number needs its constant and random references beside it
    a0, b0 = schedules(False)
    out['baselines'] = {'pool_a': C.baselines(EVAL, 'test', a0), 'pool_b': C.baselines(EVAL, 'test', b0)}
    print("baselines:", json.dumps({k: {'always_true': v['always_true']['mean'],
                                        'always_false': v['always_false']['mean'],
                                        'random': v['uniform_random']['mean'],
                                        'best_constant': v['majority_constant']}
                                    for k, v in out['baselines'].items()}, indent=2), flush=True)
    # discrete reference on one representative condition
    a, b = schedules(False)
    torch.manual_seed(0)
    trainer = build('lookup', a, 0, 8)
    print("== enumerating the lookup scaffold ==", flush=True)
    out['enumeration_lookup_ab'] = enumerate_policy(trainer, a, b)
    print(json.dumps(out['enumeration_lookup_ab'], indent=2), flush=True)
    (HERE / 'out').mkdir(exist_ok=True)
    (HERE / 'out' / 'results.json').write_text(json.dumps(out, indent=2, sort_keys=True))
    print("\n=== summary (mean over seeds, chance = 2.0/4) ===")
    for key, value in out.items():
        if not isinstance(value, dict) or 'rows' not in value: continue
        rows = value['rows']
        seen = statistics.mean(r['seen'] for r in rows); unseen = statistics.mean(r['unseen'] for r in rows)
        chose = sum(1 for r in rows if r['interpreter_selected'])
        print(f"{key:18s} seen {seen:.2f}  unseen {unseen:.2f}  interpreter chosen {chose}/{len(rows)}")

if __name__ == '__main__':
    main()
