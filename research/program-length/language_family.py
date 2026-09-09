"""Q1 on the language artifact: 10 conforming programs, and what ranking picks.

`research/language-capability/stage_b.json` records `ranked_by: order` over a
45,375-program space that was exhausted and found **10** conforming programs.
That is the one shipped artifact where a preference for shorter programs has
more than one candidate to choose between, so it is the live test of the brief's
Q1: does ranking produce a shorter program, and does it cost task quality?

STREAM, DECLARED.  `research/FINDINGS.md` section 39 established that section
19's language number reproduces only on the pre-audit stream, which
`hardening='none'` preserves bit-identically, and that the post-audit default
stream produces no length-2/4/6 episodes at all -- so the track's own training
split is empty on it.  This script pins `hardening='none'` explicitly, exactly as
section 39's reproduction does, so the "before" column is the shipped artifact
rather than an out-of-distribution transfer measurement.  Every accuracy is
reported beside its majority-constant and random baselines.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(ROOT / 'research' / 'language-capability'), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tcn.generation import Host                                   # noqa: E402

_orig = Host.create


def _pinned(generator, **kw):
    cfg = dict(kw.get('configuration') or {})
    cfg.setdefault('hardening', 'none')
    kw['configuration'] = cfg
    return _orig(generator, **kw)


Host.create = _pinned

import common                                                     # noqa: E402
import scaffolds                                                  # noqa: E402
from run_stage_b import accuracy, baselines, build_module, examples  # noqa: E402
from measure import compiled, dump, static                        # noqa: E402
from tcn.search import evaluate, program_cost, space_size         # noqa: E402

TOL = 1e-6


def conforming_set(program, rows, signals, registry, tolerance=TOL):
    """Every conforming selection, by the same `tcn.search.evaluate` a sweep uses.

    `enumerate_fit` returns one program and a count; ranking needs the set, and
    building it here keeps the scoring function identical to the shipped one.
    """
    names = [n.name for n in program.nodes]
    counts = [range(len(n.candidates)) for n in program.nodes]
    total = space_size(program)
    t0 = time.perf_counter()
    found, evaluated = [], 0
    for combination in itertools.product(*counts):
        evaluated += 1
        s = dict(zip(names, combination))
        e = evaluate(program, s, rows, signals, registry, tolerance)
        if e is not None and e <= tolerance:
            found.append(s)
    return {'space_size': total, 'evaluated': evaluated, 'exhausted': evaluated >= total,
            'count': len(found), 'unique': len(found) == 1,
            'seconds': time.perf_counter() - t0}, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-lengths', type=int, nargs='*', default=[2, 4, 6])
    ap.add_argument('--n-train', type=int, default=24)
    ap.add_argument('--tag', default='language_family')
    a = ap.parse_args()

    module, registry, frozen = build_module()
    program, signals = scaffolds.stage_b(module, registry)
    shipped = json.loads((ROOT / 'research/language-capability/stage_b.json').read_text())
    print('stage B nodes', len(program.nodes), 'space', space_size(program),
          'module', module, 'matches shipped', module == shipped['module'])

    pool = common.dataset(900, seed0=0, split='train')
    train_eps = [e for e in pool if e['length'] in a.train_lengths][:a.n_train]
    seen_held = [e for e in pool if e['length'] in a.train_lengths][a.n_train:a.n_train + 120]
    test_pool = common.dataset(500, seed0=100000, split='test')
    unseen = [e for e in test_pool if e['length'] not in a.train_lengths]
    print('train', len(train_eps), 'seen-held', len(seen_held), 'unseen', len(unseen),
          'unseen lengths', sorted({e['length'] for e in unseen}))
    tr = examples(train_eps)

    walk, found = conforming_set(program, tr, signals, registry)
    print('enumeration', walk)

    # native cases for the compiled arm: the same held-out episodes, decoded
    cases = [{'text': e['text'].decoded} for e in unseen[:24]]

    rows = []
    for s in found:
        hardened = program.harden(s)
        st = static(hardened, registry)
        bits, cost = program_cost(program, s, registry)
        comp, outs, _ = compiled(hardened, registry, cases)
        row = {'selections': {k: s[k] for k in ('symbols', 'plus', 'minus', 'answer')},
               'spelling': {n.name: [n.candidates[0].operator.name, list(n.candidates[0].sources)]
                            for n in hardened.pruned().nodes
                            if n.name in ('symbols', 'plus', 'minus', 'answer')},
               **st, 'ranked_description_bits': bits, 'ranked_execution_cost': cost,
               'bytecodes': comp['bytecodes'], 'bytecodes_total': comp['bytecodes_total'],
               'source_bytes': comp['source_bytes'],
               'outputs': [bool(o['answer']) for o in outs]}
        for tag, eps in (('train', train_eps), ('heldout_seen_lengths', seen_held),
                         ('heldout_unseen_lengths', unseen)):
            acc, per_len = accuracy(program, s, registry, eps)
            row[tag] = {'accuracy': acc, 'per_length': per_len, **baselines(eps)}
        rows.append(row)
        print(' ', row['selections'], 'nodes', row['nodes'], 'bits', bits, 'cost', cost,
              'bytecodes', row['bytecodes_total'],
              'unseen acc', round(row['heldout_unseen_lengths']['accuracy'], 4))

    def pick(key):
        return min(range(len(rows)), key=lambda i: key(rows[i])) if rows else None

    picks = {'order': 0 if rows else None,
             'description': pick(lambda r: (r['ranked_description_bits'], r['ranked_execution_cost'])),
             'cost': pick(lambda r: (r['ranked_execution_cost'], r['ranked_description_bits'])),
             'bytecodes': pick(lambda r: r['bytecodes_total'])}

    out = {'stream': "hardening='none' (pre-audit; see FINDINGS section 39)",
           'module': module, 'module_matches_shipped': module == shipped['module'],
           'scaffold_nodes': len(program.nodes), 'space_size': space_size(program),
           'enumeration': walk,
           'shipped_ranked_by': shipped['enumeration']['ranked_by'],
           'shipped_conforming': shipped['enumeration']['conforming'],
           'shipped_description_bits': shipped['enumeration']['description_bits'],
           'shipped_execution_cost': shipped['enumeration']['execution_cost'],
           'distinct_nodes': sorted({r['nodes'] for r in rows}),
           'distinct_description_bits': sorted({r['ranked_description_bits'] for r in rows}),
           'distinct_execution_cost': sorted({r['ranked_execution_cost'] for r in rows}),
           'distinct_bytecodes': sorted({r['bytecodes_total'] for r in rows}),
           'distinct_unseen_accuracy': sorted({r['heldout_unseen_lengths']['accuracy'] for r in rows}),
           'rank_picks': picks,
           'outputs_identical_across_conforming': len({tuple(r['outputs']) for r in rows}) <= 1,
           'train_episodes': len(train_eps), 'unseen_episodes': len(unseen),
           'rows': rows}
    print('distinct nodes', out['distinct_nodes'], 'bits', out['distinct_description_bits'],
          'cost', out['distinct_execution_cost'], 'bytecodes', out['distinct_bytecodes'])
    print('picks', picks)
    dump(a.tag, out)


if __name__ == '__main__':
    main()
