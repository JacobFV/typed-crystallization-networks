"""Stage A: learn the lexical perception module from raw prompt bytes.

Training episodes are restricted to short strings (L in {2,4,6}); the held-out
set is long strings (L >= 8) and unseen seeds. Positions are supervised from the
privileged `construction` latent -- the dense intermediate probe of section 7.
The agent input is `text` only.
"""
from __future__ import annotations
import sys, os, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common, scaffolds
from tcn.types import Value, BOOL
from tcn.search import enumerate_fit, space_size
from tcn.operators import Registry

def examples(eps, positions=None):
    out = []
    for e in eps:
        rng = range(len(e['string'])) if positions is None else positions
        for p in rng:
            if p >= scaffolds.CAPACITY: continue
            bit = p < len(e['string']) and e['string'][p] == '('
            out.append({'inputs': {'text': e['text'], 'pos': Value.of(scaffolds.POS, p)},
                        'targets': {'open': Value.of(BOOL, bool(bit))}})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-lengths', type=int, nargs='*', default=[2, 4, 6])
    ap.add_argument('--n-train', type=int, default=12)
    ap.add_argument('--n-test', type=int, default=40)
    ap.add_argument('--out', default=os.path.join(HERE, 'stage_a.json'))
    a = ap.parse_args()

    pool = common.dataset(600, seed0=0, split='train')
    train_eps = [e for e in pool if e['length'] in a.train_lengths][:a.n_train]
    heldout_seen = [e for e in pool if e['length'] in a.train_lengths][a.n_train:a.n_train + 40]
    pool_t = common.dataset(400, seed0=100000, split='test')
    heldout_unseen = [e for e in pool_t if e['length'] not in a.train_lengths][:a.n_test]

    program, registry, signals = scaffolds.stage_a()
    tr = examples(train_eps)
    print(f'stage A: {len(train_eps)} episodes (lengths {sorted({e["length"] for e in train_eps})}), '
          f'{len(tr)} supervised positions, space {space_size(program)}')

    t0 = time.perf_counter()
    res = enumerate_fit(program, tr, signals, registry, tolerance=1e-6, max_programs=1 << 20)
    print('enumeration:', json.dumps(res.to_dict(), indent=1))

    report = {'space_size': space_size(program), 'train_episodes': len(train_eps),
              'train_positions': len(tr), 'train_lengths': sorted({e['length'] for e in train_eps}),
              'enumeration': res.to_dict()}
    if res.solved:
        chosen = program.harden(res.selections)
        sel = {n.name: n.candidates[0].to_dict() for n in chosen.nodes}
        report['selected'] = {'base': sel['base']['sources'][0], 'open_byte': sel['open']['sources'][1]}
        for name, eps in (('heldout_seen_lengths', heldout_seen), ('heldout_unseen_lengths', heldout_unseen)):
            ex = examples(eps)
            ok = 0
            for x in ex:
                out, _ = chosen.run(x['inputs'], registry=registry)
                ok += out['open'].decoded == x['targets']['open'].decoded
            report[name] = {'episodes': len(eps), 'positions': len(ex), 'accuracy': ok / max(1, len(ex)),
                            'lengths': sorted({e['length'] for e in eps})}
            print(name, report[name])
        report['module_program'] = chosen.pruned().to_dict()
    json.dump(report, open(a.out, 'w'), indent=1)

if __name__ == '__main__':
    main()
