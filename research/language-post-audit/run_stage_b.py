"""Stage B on the post-audit stream, with stage A frozen.

Scaffold and search are the track's own, imported unmodified from
`research/language-capability/scaffolds.py` and `tcn.search`. What changes is
the episode source (`hardening` pinned explicitly) and the split (train lengths
{10,12,14}, held out {16,18,20,22}).

`--positions` is the scaffold's own capacity parameter. The pre-audit run used
16, which covered its longest string (14). Post-audit strings run to 22, so 16
no longer covers the string; both arms are run and both are reported.
"""
from __future__ import annotations
import sys, os, json, time, argparse, collections, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
import common, splits as splits_mod

_spec = importlib.util.spec_from_file_location(
    'lc_scaffolds', os.path.join(ROOT, 'research', 'language-capability', 'scaffolds.py'))
scaffolds = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(scaffolds)

from tcn.types import Value, BOOL
from tcn.search import enumerate_fit, space_size


def build_module(registry=None):
    """The stage-A selection, hardened and frozen. Verified by run_stage_a.py on
    this stream: base=14, open_byte=40, certificate 'unique', 1.000 held-out."""
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
    ap.add_argument('--positions', type=int, default=22)
    ap.add_argument('--hardening', default=common.STREAM_POST_AUDIT)
    ap.add_argument('--n-train', type=int, default=24)
    ap.add_argument('--out', default=None)
    ap.add_argument('--rank', default='order')
    a = ap.parse_args()
    out_path = a.out or os.path.join(HERE, f'stage_b_p{a.positions}.json')

    module, registry, frozen = build_module()
    program, signals = scaffolds.stage_b(module, registry, positions=a.positions)
    print(f'stage B [{a.hardening}] positions={a.positions} nodes {len(program.nodes)} '
          f'space {space_size(program)}')

    s = splits_mod.build(hardening=a.hardening, n_train=a.n_train)
    train_eps, seen_held, unseen = s['train'], s['heldout_seen_lengths'], s['heldout_unseen_lengths']
    tr = examples(train_eps)

    t0 = time.perf_counter()
    res = enumerate_fit(program, tr, signals, registry, tolerance=1e-6, rank=a.rank)
    print('enumeration:', json.dumps({k: v for k, v in res.to_dict().items()
                                      if k != 'selections'}, indent=1))

    report = {'stream': a.hardening, 'positions': a.positions,
              'module': module, 'module_cost': frozen.execution_cost(registry),
              'nodes': len(program.nodes), 'space_size': space_size(program),
              'train_episodes': len(train_eps),
              'train_lengths': sorted({e['length'] for e in train_eps}),
              'train_depths': sorted({e['depth'] for e in train_eps}),
              'seconds': time.perf_counter() - t0,
              'enumeration': res.to_dict()}
    if res.solved:
        sel = res.selections
        chosen = program.harden(sel)
        report['selected'] = {n.name: [n.candidates[0].operator.name, list(n.candidates[0].sources)]
                              for n in chosen.pruned().nodes
                              if n.name in ('symbols', 'plus', 'minus', 'answer')}
        report['description_bits'] = chosen.pruned().description_bits(registry)
        report['execution_cost'] = chosen.pruned().execution_cost(registry)
        for name, eps in (('train', train_eps), ('heldout_seen_lengths', seen_held),
                          ('heldout_unseen_lengths', unseen)):
            acc, per_len = accuracy(program, sel, registry, eps)
            report[name] = {'accuracy': acc, 'per_length': per_len,
                            'lengths': sorted({e['length'] for e in eps}),
                            'depths': sorted({e['depth'] for e in eps}), **baselines(eps)}
            print(name, round(acc, 4), 'majority', round(report[name]['majority_constant'], 4), per_len)
        report['program'] = chosen.pruned().to_dict()
    else:
        # no conforming program: record what each split's majority is anyway
        for name, eps in (('train', train_eps), ('heldout_seen_lengths', seen_held),
                          ('heldout_unseen_lengths', unseen)):
            report[name] = {'accuracy': None, **baselines(eps),
                            'lengths': sorted({e['length'] for e in eps}),
                            'depths': sorted({e['depth'] for e in eps})}
    json.dump(report, open(out_path, 'w'), indent=1)
    print('wrote', out_path)


if __name__ == '__main__':
    main()
