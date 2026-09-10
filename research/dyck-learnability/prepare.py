"""Draw the split once, pickle it, so every arm in this track sees identical episodes.

The split is section 45's, re-posed by the same vendored `splits.py`: train
lengths {10,12,14}, held out {16,18,20,22}, `hardening` pinned explicitly to
`context_free_language` (the post-audit stream) at every draw.

Nothing here is tuned. The pickle exists only so eleven parallel enumeration
shards and the gradient arms cannot silently disagree about their episodes.
"""
from __future__ import annotations
import sys, os, json, pickle, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
import common, splits as splits_mod

OUT = os.path.join(HERE, 'out')
PKL = os.path.join(OUT, 'splits.pkl')


def load():
    with open(PKL, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()
    s = splits_mod.build(hardening=common.STREAM_POST_AUDIT)
    with open(PKL, 'wb') as f:
        pickle.dump(s, f)
    d = splits_mod.describe(s)
    d['seconds'] = time.perf_counter() - t0
    d['stream_pinned'] = common.STREAM_POST_AUDIT
    json.dump(d, open(os.path.join(HERE, 'splits.json'), 'w'), indent=1)
    for k, v in d.items():
        if isinstance(v, dict):
            print(k, {kk: v[kk] for kk in ('n', 'per_length', 'majority_constant',
                                           'fraction_string_seen_in_training') if kk in v})
    r = load()
    same = all(len(r[k]) == len(s[k]) and
               all(a['seed'] == b['seed'] and a['string'] == b['string']
                   for a, b in zip(r[k], s[k])) for k in s)
    print('seconds', round(d['seconds'], 1), '| pickle round-trip identical:', same)
