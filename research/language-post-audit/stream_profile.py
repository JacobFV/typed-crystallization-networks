"""What the post-audit stream actually emits, and what the pre-audit one did.

Nothing here is a result about the method; it is the distribution profile that
any split has to be designed against. Every draw pins `hardening` explicitly.
"""
from __future__ import annotations
import sys, os, json, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common

N = 1200


def profile(hardening, tag, n=N, seed0=0, split='train'):
    eps = common.dataset(n, seed0=seed0, split=split, hardening=hardening)
    by_len = collections.Counter(e['length'] for e in eps)
    by_depth = collections.Counter(e['depth'] for e in eps)
    yes = sum(e['label'] for e in eps)
    counts_ok = sum(common.counts_match(e['string']) for e in eps)
    # the counting oracle: predict 'balanced' iff #( == #)
    count_oracle = sum(common.counts_match(e['string']) == e['label'] for e in eps)
    dyck_oracle = sum(common.balanced(e['string']) == e['label'] for e in eps)
    # label as a function of length / depth alone
    len_tab = collections.defaultdict(collections.Counter)
    for e in eps:
        len_tab[e['length']][e['label']] += 1
    return {
        'stream': tag, 'hardening_arg': hardening, 'n': n, 'split': split, 'seed0': seed0,
        'lengths': dict(sorted(by_len.items())),
        'depths': dict(sorted(by_depth.items())),
        'positive_rate': yes / n,
        'majority_constant': max(yes, n - yes) / n,
        'counts_match_rate': counts_ok / n,
        'counting_oracle_accuracy': count_oracle / n,
        'dyck_oracle_accuracy': dyck_oracle / n,
        'distinct_strings': len({e['string'] for e in eps}),
        'distinct_prompts': len({e['prompt'] for e in eps}),
        'label_by_length': {k: {'n': sum(v.values()), 'positive_rate': v[True] / sum(v.values())}
                            for k, v in sorted(len_tab.items())},
        'min_prefix_hist': dict(sorted(collections.Counter(
            common.min_prefix(e['string']) for e in eps).items())),
    }


if __name__ == '__main__':
    out = {
        'post_audit_train': profile(common.STREAM_POST_AUDIT, 'post-audit (hardened)'),
        'post_audit_test': profile(common.STREAM_POST_AUDIT, 'post-audit (hardened)',
                                   n=800, seed0=100000, split='test'),
        'pre_audit_train': profile(common.STREAM_PRE_AUDIT, "pre-audit (hardening='none')"),
    }
    json.dump(out, open(os.path.join(HERE, 'stream_profile.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))
