"""Matched-information baselines on the same held-out episodes.

Each is the accuracy of the *optimal* deterministic predictor from a stated
feature, fitted on the training episodes and read out on the held-out ones -- an
upper bound on anything a program could do from that feature alone.
"""
from __future__ import annotations
import sys, os, json, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common

FEATURES = {
    'constant': lambda e: 0,
    'length': lambda e: e['length'],
    'first_symbol': lambda e: e['string'][0],
    'last_symbol': lambda e: e['string'][-1],
    'first_and_last': lambda e: (e['string'][0], e['string'][-1]),
    'bigram_multiset': lambda e: tuple(sorted(collections.Counter(
        e['string'][i:i + 2] for i in range(len(e['string']) - 1)).items())),
    'open_count_and_length': lambda e: (e['string'].count('('), e['length']),
}

def fit_predict(train, test, feature):
    table = collections.defaultdict(collections.Counter)
    for e in train: table[feature(e)][e['label']] += 1
    prior = collections.Counter(e['label'] for e in train).most_common(1)[0][0]
    ok = 0
    for e in test:
        c = table.get(feature(e))
        ok += (c.most_common(1)[0][0] if c else prior) == e['label']
    return ok / max(1, len(test))

def main():
    pool = common.dataset(900, seed0=0, split='train')
    train = [e for e in pool if e['length'] in (2, 4, 6)][:24]
    seen = [e for e in pool if e['length'] in (2, 4, 6)][24:144]
    test_pool = common.dataset(1500, seed0=100000, split='test')
    unseen = [e for e in test_pool if e['length'] not in (2, 4, 6)]
    out = {}
    for split, eps in (('train', train), ('heldout_seen_lengths', seen), ('heldout_unseen_lengths', unseen)):
        yes = sum(e['label'] for e in eps)
        out[split] = {'n': len(eps), 'positive_rate': yes / len(eps),
                      'majority_constant': max(yes, len(eps) - yes) / len(eps), 'random': 0.5,
                      'lengths': sorted({e['length'] for e in eps}),
                      'depths': sorted({e['depth'] for e in eps}),
                      'fitted_from': {name: fit_predict(train, eps, f) for name, f in FEATURES.items()}}
    json.dump(out, open(os.path.join(HERE, 'baselines.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == '__main__':
    main()
