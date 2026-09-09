"""Every conforming stage-B program, and what each of them does on held-out structure.

`enumerate_fit` returns one program and a count. The standing rule is that
non-uniqueness is the norm and a conforming program is not necessarily the right
one (findings section 14), so this sweep keeps the whole conforming set and
scores each member on held-out lengths it never saw.
"""
from __future__ import annotations
import sys, os, json, time, itertools, argparse, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common, scaffolds
from run_stage_b import build_module, examples, accuracy, baselines
from tcn.search import evaluate, space_size, candidate_counts

def sweep(program, ex, signals, registry, tolerance=1e-6):
    counts = candidate_counts(program); names = [n.name for n in program.nodes]
    found = []; evaluated = 0; t0 = time.perf_counter()
    for combo in itertools.product(*(range(c) for c in counts)):
        sel = dict(zip(names, combo)); evaluated += 1
        err = evaluate(program, sel, ex, signals, registry, tolerance)
        if err is not None and err <= tolerance: found.append(sel)
    return found, evaluated, time.perf_counter() - t0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-lengths', type=int, nargs='*', default=[2, 4, 6])
    ap.add_argument('--n-train', type=int, default=24)
    ap.add_argument('--out', default=os.path.join(HERE, 'conforming.json'))
    a = ap.parse_args()
    module, registry, frozen = build_module()
    program, signals = scaffolds.stage_b(module, registry)
    pool = common.dataset(900, seed0=0, split='train')
    train_eps = [e for e in pool if e['length'] in a.train_lengths][:a.n_train]
    seen_held = [e for e in pool if e['length'] in a.train_lengths][a.n_train:a.n_train + 120]
    test_pool = common.dataset(1500, seed0=100000, split='test')
    unseen = [e for e in test_pool if e['length'] not in a.train_lengths]
    found, evaluated, secs = sweep(program, examples(train_eps), signals, registry)
    print(f'conforming {len(found)} / {evaluated} in {secs:.1f}s')
    rows = []
    for sel in found:
        seen_acc, _ = accuracy(program, sel, registry, seen_held)
        uns_acc, per_len = accuracy(program, sel, registry, unseen)
        rows.append({'selection': {k: sel[k] for k in ('symbols', 'plus', 'minus', 'answer')},
                     'chosen': {'symbols_sub': program.nodes[1].candidates[sel['symbols']].sources[1],
                                'plus': program.nodes[2].candidates[sel['plus']].sources[0],
                                'minus': program.nodes[3].candidates[sel['minus']].sources[0],
                                'rule': [program.nodes[-1].candidates[sel['answer']].operator.name,
                                         program.nodes[-1].candidates[sel['answer']].sources[1]]},
                     'heldout_seen_lengths': seen_acc, 'heldout_unseen_lengths': uns_acc,
                     'per_length': per_len})
    rows.sort(key=lambda r: -r['heldout_unseen_lengths'])
    report = {'space_size': space_size(program), 'evaluated': evaluated, 'seconds': secs,
              'conforming': len(found), 'unique': len(found) == 1,
              'train_episodes': len(train_eps), 'train_lengths': sorted({e['length'] for e in train_eps}),
              'baselines': {'heldout_seen_lengths': baselines(seen_held),
                            'heldout_unseen_lengths': baselines(unseen)},
              'unseen_accuracy_distribution': collections.Counter(round(r['heldout_unseen_lengths'], 3) for r in rows),
              'programs': rows}
    report['unseen_accuracy_distribution'] = dict(sorted(report['unseen_accuracy_distribution'].items()))
    json.dump(report, open(a.out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != 'programs'}, indent=1))
    for r in rows[:5]: print(r['chosen'], r['heldout_seen_lengths'], r['heldout_unseen_lengths'])
    for r in rows[-3:]: print(r['chosen'], r['heldout_seen_lengths'], r['heldout_unseen_lengths'])

if __name__ == '__main__':
    main()
