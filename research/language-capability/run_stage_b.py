"""Stage B: grammaticality from the recovered symbols, with stage A frozen.

The stage-A program is hardened, pruned and registered as a frozen module -- an
immutable callable operator with no internal gradients (ARCHITECTURE section 4).
Stage B calls it once per position and searches only the length relation, the
symbol-to-step map, and the rule that reads the accumulated count.

Training is restricted to short strings; the held-out set is longer strings,
which is a *generating structure* holdout, not a seed holdout.
"""
from __future__ import annotations
import sys, os, json, time, argparse, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common, scaffolds
from tcn.types import Value, BOOL
from tcn.search import enumerate_fit, space_size
from tcn.operators import Registry

def build_module(registry=None):
    program, registry, _ = scaffolds.stage_a(registry=registry)
    frozen = program.harden({'bytes': 0, 'base': 14, 'addr': 0, 'byte': 0, 'open': 40})
    return registry.register_module(frozen), registry, frozen

def examples(eps):
    return [{'inputs': {'text': e['text']},
             'targets': {'answer': Value.of(BOOL, bool(e['label']))}} for e in eps]

def accuracy(program, selections, registry, eps):
    ok = 0; per_len = collections.defaultdict(lambda: [0, 0])
    for e in eps:
        try:
            out, _ = program.run({'text': e['text']}, registry=registry, selections=selections)
            hit = bool(out['answer'].decoded) == bool(e['label'])
        except Exception:
            hit = False
        ok += hit; per_len[e['length']][0] += hit; per_len[e['length']][1] += 1
    return ok / max(1, len(eps)), {k: v[0] / v[1] for k, v in sorted(per_len.items())}

def baselines(eps):
    yes = sum(e['label'] for e in eps); n = len(eps)
    return {'n': n, 'constant_yes': yes / n, 'constant_no': (n - yes) / n,
            'majority_constant': max(yes, n - yes) / n, 'random': 0.5}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-lengths', type=int, nargs='*', default=[2, 4, 6])
    ap.add_argument('--n-train', type=int, default=24)
    ap.add_argument('--out', default=os.path.join(HERE, 'stage_b.json'))
    ap.add_argument('--rank', default='order')
    a = ap.parse_args()
    module, registry, frozen = build_module()
    program, signals = scaffolds.stage_b(module, registry)
    print('stage B nodes', len(program.nodes), 'space', space_size(program))

    pool = common.dataset(900, seed0=0, split='train')
    train_eps = [e for e in pool if e['length'] in a.train_lengths][:a.n_train]
    seen_held = [e for e in pool if e['length'] in a.train_lengths][a.n_train:a.n_train + 120]
    test_pool = common.dataset(500, seed0=100000, split='test')
    unseen = [e for e in test_pool if e['length'] not in a.train_lengths]

    tr = examples(train_eps)
    t0 = time.perf_counter()
    res = enumerate_fit(program, tr, signals, registry, tolerance=1e-6, rank=a.rank)
    print('enumeration:', json.dumps(res.to_dict(), indent=1))
    report = {'module': module, 'module_cost': frozen.execution_cost(registry),
              'nodes': len(program.nodes), 'space_size': space_size(program),
              'train_episodes': len(train_eps), 'train_lengths': sorted({e['length'] for e in train_eps}),
              'enumeration': res.to_dict(), 'baselines': {}}
    if res.solved:
        sel = res.selections
        chosen = program.harden(sel)
        report['selected'] = {n.name: [n.candidates[0].operator.name, list(n.candidates[0].sources)]
                              for n in chosen.pruned().nodes if n.name in ('symbols', 'plus', 'minus', 'answer')}
        for name, eps in (('train', train_eps), ('heldout_seen_lengths', seen_held),
                          ('heldout_unseen_lengths', unseen)):
            acc, per_len = accuracy(program, sel, registry, eps)
            report[name] = {'accuracy': acc, 'per_length': per_len, **baselines(eps)}
            print(name, round(acc, 4), report[name]['majority_constant'], per_len)
        report['program'] = chosen.pruned().to_dict()
    json.dump(report, open(a.out, 'w'), indent=1)

if __name__ == '__main__':
    main()
