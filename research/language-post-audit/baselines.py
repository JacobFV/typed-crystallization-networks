"""Matched-information baselines on the post-audit split.

The feature set is the original track's, imported verbatim from
`research/language-capability/baselines.py` so that no baseline is quietly
weakened when the stream changes. Each number is the accuracy of the *optimal*
deterministic predictor from that feature, fitted on the 24 training episodes
and read out on the held-out ones -- an upper bound on anything a program could
do from that feature alone.

Reported beside them: the majority constant, random (0.5), and the two oracles
that section 43 named -- the counting oracle (`#( == #)`) and the Dyck oracle.
"""
from __future__ import annotations
import sys, os, json, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
import common, splits as splits_mod

_spec = importlib.util.spec_from_file_location(
    'lc_baselines', os.path.join(ROOT, 'research', 'language-capability', 'baselines.py'))
lc_base = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(lc_base)
FEATURES = lc_base.FEATURES
fit_predict = lc_base.fit_predict


def nearest_prompt(train, test):
    """The exploit section 24 re-drew the lesson to kill: copy the label of an
    identical training string, else the training prior."""
    table = {e['string']: e['label'] for e in train}
    import collections
    prior = collections.Counter(e['label'] for e in train).most_common(1)[0][0]
    return sum(table.get(e['string'], prior) == e['label'] for e in test) / max(1, len(test))


def main():
    s = splits_mod.build()
    train = s['train']
    out = {'stream': common.STREAM_POST_AUDIT,
           'train_lengths': list(splits_mod.TRAIN_LENGTHS),
           'heldout_lengths': list(splits_mod.HELDOUT_LENGTHS)}
    for split, eps in s.items():
        n = len(eps)
        yes = sum(e['label'] for e in eps)
        out[split] = {
            'n': n, 'positive_rate': yes / n,
            'majority_constant': max(yes, n - yes) / n, 'random': 0.5,
            'lengths': sorted({e['length'] for e in eps}),
            'depths': sorted({e['depth'] for e in eps}),
            'counting_oracle': sum(common.counts_match(e['string']) == e['label'] for e in eps) / n,
            'dyck_oracle': sum(common.balanced(e['string']) == e['label'] for e in eps) / n,
            'training_string_lookup': nearest_prompt(train, eps),
            'fitted_from': {name: fit_predict(train, eps, f) for name, f in FEATURES.items()},
        }
        out[split]['best_fitted_feature'] = max(out[split]['fitted_from'].items(), key=lambda kv: kv[1])
    json.dump(out, open(os.path.join(HERE, 'baselines.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
