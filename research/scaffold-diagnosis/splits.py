"""The post-audit split, stated once and reused by every script in this track.

Held out: **string length**, the generating structure of the Dyck word, not the
seed. The post-audit stream emits lengths {10,12,14,16,18,20,22} only (all even,
because the hardened draw builds both classes from a `_dyck` word of a drawn
number of pairs and the negative is a *transposition* of it). Training takes the
three shortest, held-out the four longest.

The pre-audit track held out lengths >= 8 while training on {2,4,6}; those
lengths do not exist post-audit, which is why its training split is empty and
`final_eval.py` dies. This is the same holdout re-posed onto lengths that exist.

Nothing here is tuned: this is the first and only split run (see RESULTS.md
"Splits tried").
"""
from __future__ import annotations
import sys, os, json, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common

TRAIN_LENGTHS = (10, 12, 14)
HELDOUT_LENGTHS = (16, 18, 20, 22)

N_TRAIN = 24            # same episode budget as the pre-audit track's stage B
N_SEEN_HELD = 120       # same as the pre-audit track
TRAIN_POOL = 900
TEST_POOL = 1500
TEST_SEED0 = 100000


def build(hardening=common.STREAM_POST_AUDIT, n_train=N_TRAIN, n_seen=N_SEEN_HELD,
          train_lengths=TRAIN_LENGTHS, heldout_lengths=HELDOUT_LENGTHS):
    """train / heldout_seen_lengths / heldout_unseen_lengths, all pinned to one stream.

    `train_lengths` / `heldout_lengths` are exposed only so the pre-audit control
    (`control_preaudit.py`) can re-pose the *original* split with this same code.
    """
    pool = common.dataset(TRAIN_POOL, seed0=0, split='train', hardening=hardening)
    short = [e for e in pool if e['length'] in train_lengths]
    train = short[:n_train]
    seen = short[n_train:n_train + n_seen]
    test_pool = common.dataset(TEST_POOL, seed0=TEST_SEED0, split='test', hardening=hardening)
    unseen = [e for e in test_pool if e['length'] in heldout_lengths]
    return {'train': train, 'heldout_seen_lengths': seen, 'heldout_unseen_lengths': unseen}


def describe(splits, train_key='train'):
    train_strings = {e['string'] for e in splits[train_key]}
    out = {'train_lengths': list(TRAIN_LENGTHS), 'heldout_lengths': list(HELDOUT_LENGTHS),
           'stream': splits[train_key][0]['stream'] if splits[train_key] else None}
    for name, eps in splits.items():
        n = len(eps)
        yes = sum(e['label'] for e in eps)
        out[name] = {
            'n': n,
            'per_length': dict(sorted(collections.Counter(e['length'] for e in eps).items())),
            'per_depth': dict(sorted(collections.Counter(e['depth'] for e in eps).items())),
            'lengths': sorted({e['length'] for e in eps}),
            'depths': sorted({e['depth'] for e in eps}),
            'distinct_strings': len({e['string'] for e in eps}),
            'positive_rate': yes / n if n else None,
            'majority_constant': max(yes, n - yes) / n if n else None,
            'fraction_string_seen_in_training': (
                sum(e['string'] in train_strings for e in eps) / n if n else None),
            'counts_match_rate': sum(common.counts_match(e['string']) for e in eps) / n if n else None,
        }
    return out


if __name__ == '__main__':
    s = build()
    d = describe(s)
    json.dump(d, open(os.path.join(HERE, 'splits.json'), 'w'), indent=1)
    print(json.dumps(d, indent=1))
