"""The bounding run: same method, same search, a scaffold that can express Dyck.

Answers the second question in the brief -- is the post-audit task solvable at
all by a program in this family, and if the track's own scaffold cannot reach
it, is that a search failure or an expressiveness limit.

Search is `tcn.search.enumerate_fit`, unmodified; the split is the same one
`splits.py` states; `hardening` is pinned.
"""
from __future__ import annotations
import sys, os, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
import common, splits as splits_mod
from run_stage_b import build_module, examples, accuracy, baselines
from dyck_scaffold import stage_b_dyck
from tcn.search import enumerate_fit, space_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--positions', type=int, default=22)
    ap.add_argument('--hardening', default=common.STREAM_POST_AUDIT)
    ap.add_argument('--n-train', type=int, default=24)
    ap.add_argument('--rank', default='order')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out_path = a.out or os.path.join(HERE, f'dyck_p{a.positions}.json')

    module, registry, frozen = build_module()
    program, signals = stage_b_dyck(module, registry, positions=a.positions)
    print(f'dyck scaffold [{a.hardening}] positions={a.positions} nodes {len(program.nodes)} '
          f'space {space_size(program)}', flush=True)

    s = splits_mod.build(hardening=a.hardening, n_train=a.n_train)
    train_eps, seen_held, unseen = s['train'], s['heldout_seen_lengths'], s['heldout_unseen_lengths']
    tr = examples(train_eps)

    t0 = time.perf_counter()
    res = enumerate_fit(program, tr, signals, registry, tolerance=1e-6, rank=a.rank)
    print('enumeration:', json.dumps({k: v for k, v in res.to_dict().items()
                                      if k != 'selections'}, indent=1), flush=True)

    report = {'stream': a.hardening, 'positions': a.positions, 'scaffold': 'stage_b_dyck',
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
                              if n.name in ('symbols', 'plus', 'minus', 'total_ok', 'min_ok', 'answer')}
        report['description_bits'] = chosen.pruned().description_bits(registry)
        report['execution_cost'] = chosen.pruned().execution_cost(registry)
        report['live_nodes'] = len(chosen.pruned().nodes)
        for name, eps in (('train', train_eps), ('heldout_seen_lengths', seen_held),
                          ('heldout_unseen_lengths', unseen)):
            acc, per_len = accuracy(program, sel, registry, eps)
            report[name] = {'accuracy': acc, 'per_length': per_len,
                            'lengths': sorted({e['length'] for e in eps}),
                            'depths': sorted({e['depth'] for e in eps}), **baselines(eps)}
            print(name, round(acc, 4), 'majority', round(report[name]['majority_constant'], 4),
                  per_len, flush=True)
        exported = chosen.pruned()
        t1 = time.perf_counter(); n = min(50, len(unseen))
        for e in unseen[:n]:
            exported.run({'text': e['text']}, registry=registry)
        report['batch_one_latency_ms'] = (time.perf_counter() - t1) / n * 1e3
        report['program'] = exported.to_dict()
    else:
        for name, eps in (('train', train_eps), ('heldout_seen_lengths', seen_held),
                          ('heldout_unseen_lengths', unseen)):
            report[name] = {'accuracy': None, **baselines(eps)}
    json.dump(report, open(out_path, 'w'), indent=1)
    print('wrote', out_path)


if __name__ == '__main__':
    main()
