"""Control: this track's own code, run on the PRE-audit stream and the ORIGINAL split.

If the two-stage method fails on the post-audit stream, the failure has to be
attributable to the stream and not to this re-implementation. This script runs
the identical code path -- same `common.episode`, same imported scaffolds, same
`enumerate_fit` -- with `hardening='none'` and train lengths {2,4,6}, and should
reproduce FINDINGS section 19 / `research/language-capability/stage_b.json`
(1.000 train, 1.000 held-out unseen lengths, 45,375 space, 10 conforming).
"""
from __future__ import annotations
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
import common, splits as splits_mod
from run_stage_b import build_module, examples, accuracy, baselines, scaffolds
from tcn.search import enumerate_fit, space_size

TRAIN_LENGTHS = (2, 4, 6)
HELDOUT_LENGTHS = (8, 10, 12, 14, 16, 18, 20, 22)


def main():
    module, registry, frozen = build_module()
    program, signals = scaffolds.stage_b(module, registry, positions=16)
    s = splits_mod.build(hardening=common.STREAM_PRE_AUDIT, n_train=24,
                         train_lengths=TRAIN_LENGTHS, heldout_lengths=HELDOUT_LENGTHS)
    tr = examples(s['train'])
    print(f"control [hardening='none'] positions=16 nodes {len(program.nodes)} "
          f"space {space_size(program)} train {len(s['train'])}", flush=True)
    t0 = time.perf_counter()
    res = enumerate_fit(program, tr, signals, registry, tolerance=1e-6)
    report = {'stream': common.STREAM_PRE_AUDIT, 'positions': 16,
              'train_lengths': sorted({e['length'] for e in s['train']}),
              'space_size': space_size(program), 'seconds': time.perf_counter() - t0,
              'enumeration': res.to_dict()}
    print('enumeration:', json.dumps({k: v for k, v in res.to_dict().items()
                                      if k != 'selections'}, indent=1), flush=True)
    if res.solved:
        for name, eps in (('train', s['train']),
                          ('heldout_seen_lengths', s['heldout_seen_lengths']),
                          ('heldout_unseen_lengths', s['heldout_unseen_lengths'])):
            acc, per_len = accuracy(program, res.selections, registry, eps)
            report[name] = {'accuracy': acc, 'per_length': per_len, **baselines(eps)}
            print(name, round(acc, 6), 'majority', round(report[name]['majority_constant'], 4),
                  per_len, flush=True)
    json.dump(report, open(os.path.join(HERE, 'control_preaudit.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
