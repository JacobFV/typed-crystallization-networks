"""Stage A on the post-audit stream: lexical perception from raw prompt bytes.

The scaffold is `research/language-capability/scaffolds.py:stage_a`, imported
verbatim and unmodified; the search is `tcn.search.enumerate_fit`, unmodified.
Only the episode source changes: `common.dataset` here pins `hardening`
explicitly and the split holds out lengths {16,18,20,22} instead of {8,10,12,14}
because the pre-audit lengths {2,4,6} do not occur post-audit.

Positions are supervised from the privileged `construction` latent (a probe --
supervision only). The agent input is `text` and `pos` only.
"""
from __future__ import annotations
import sys, os, json, time, argparse, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import common, splits as splits_mod

# The track's own scaffolds, loaded from research/language-capability unchanged.
_spec = importlib.util.spec_from_file_location(
    'lc_scaffolds', os.path.join(ROOT, 'research', 'language-capability', 'scaffolds.py'))
scaffolds = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(scaffolds)

from tcn.types import Value, BOOL
from tcn.search import enumerate_fit, space_size


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
    ap.add_argument('--n-train', type=int, default=12)
    ap.add_argument('--n-test', type=int, default=40)
    ap.add_argument('--hardening', default=common.STREAM_POST_AUDIT)
    ap.add_argument('--out', default=os.path.join(HERE, 'stage_a.json'))
    a = ap.parse_args()

    train_lengths = list(splits_mod.TRAIN_LENGTHS)
    pool = common.dataset(600, seed0=0, split='train', hardening=a.hardening)
    short = [e for e in pool if e['length'] in train_lengths]
    train_eps = short[:a.n_train]
    heldout_seen = short[a.n_train:a.n_train + 40]
    pool_t = common.dataset(400, seed0=splits_mod.TEST_SEED0, split='test', hardening=a.hardening)
    heldout_unseen = [e for e in pool_t if e['length'] in splits_mod.HELDOUT_LENGTHS][:a.n_test]

    program, registry, signals = scaffolds.stage_a()
    tr = examples(train_eps)
    print(f'stage A [{a.hardening}]: {len(train_eps)} episodes '
          f'(lengths {sorted({e["length"] for e in train_eps})}), '
          f'{len(tr)} supervised positions, space {space_size(program)}')

    t0 = time.perf_counter()
    res = enumerate_fit(program, tr, signals, registry, tolerance=1e-6, max_programs=1 << 20)
    print('enumeration:', json.dumps(res.to_dict(), indent=1))

    report = {'stream': a.hardening, 'space_size': space_size(program),
              'train_episodes': len(train_eps), 'train_positions': len(tr),
              'train_lengths': sorted({e['length'] for e in train_eps}),
              'train_depths': sorted({e['depth'] for e in train_eps}),
              'seconds': time.perf_counter() - t0,
              'enumeration': res.to_dict()}
    if res.solved:
        chosen = program.harden(res.selections)
        sel = {n.name: n.candidates[0].to_dict() for n in chosen.nodes}
        report['selected'] = {'base': sel['base']['sources'][0], 'open_byte': sel['open']['sources'][1]}
        for name, eps in (('heldout_seen_lengths', heldout_seen),
                          ('heldout_unseen_lengths', heldout_unseen)):
            ex = examples(eps)
            ok = 0
            for x in ex:
                out, _ = chosen.run(x['inputs'], registry=registry)
                ok += out['open'].decoded == x['targets']['open'].decoded
            report[name] = {'episodes': len(eps), 'positions': len(ex),
                            'accuracy': ok / max(1, len(ex)),
                            'lengths': sorted({e['length'] for e in eps}),
                            'depths': sorted({e['depth'] for e in eps})}
            print(name, report[name])
        report['module_program'] = chosen.pruned().to_dict()
    json.dump(report, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
