"""Q1: the unwindowed enumeration over the min-prefix scaffold, run as an exhausted union.

Section 45 recovered the Dyck program with the **unmodified** search, but only
inside a hand-chosen window `c in [95,110)` -- a window picked from knowledge of
the answer. Its attempt at the full space (680,625 programs, measured 4.18 h) was
stopped at ~52% and therefore carried no certificate.

This runs the whole space. The search is `tcn.search.enumerate_fit`, unmodified,
and the scaffold is `dyck_scaffold.stage_b_dyck`, unmodified. The only thing this
script does is **partition** the `c` axis into contiguous shards so the pieces can
run concurrently: `sub_range=range(lo,hi)` is exactly the argument section 45's
own `run_dyck.py` exposes, and the shards are disjoint and cover `[0,121)`, so
the union of eleven exhausted shards is an exhaustive decision of the full space.

That partition is order-preserving, which is what makes the "where does the first
conforming member sit" question answerable. `enumerate_fit` walks
`itertools.product` over candidate counts in node order, and in this scaffold the
`symbols` node (the one carrying `c`) is the most significant multi-candidate
node, with `plus`, `minus`, `total_ok`, `min_ok` after it. So the global mixed-radix
index of a member is

    index = (((c_index*5 + plus)*5 + minus)*15 + total_ok)*15 + min_ok

and a contiguous window of `c` is a contiguous block of the global enumeration
order. Verified against section 45's own witness index (571,822) by
`q1_aggregate.py`.

`rank='order'` is used deliberately: it returns the *first* conforming member in
enumeration order, which is the thing Q1 asks about. Ranking by description bits
would answer a different question ("what is the best program") and would hide
where the search first succeeds.

`hardening` is pinned by `common.STREAM_POST_AUDIT` inside `prepare.py`; every
number this script produces comes from the post-audit
`hardening='context_free_language'` stream.
"""
from __future__ import annotations
import sys, os, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
import common, prepare
from run_stage_b import build_module, examples, accuracy, baselines
from dyck_scaffold import stage_b_dyck
from tcn.search import enumerate_fit, space_size, candidate_counts

STEPS = (-2, -1, 0, 1, 2)
RULES = [(op, v) for v in (-2, -1, 0, 1, 2) for op in ('eq', 'ge', 'le')]


def global_index(program, sub_lo, selections):
    """Mixed-radix index of a selection in the FULL (c in [0,121)) enumeration order."""
    c_index = sub_lo + selections['symbols']
    return ((((c_index * 5 + selections['plus']) * 5 + selections['minus']) * 15
             + selections['total_ok']) * 15 + selections['min_ok'])


def decode(program, sub_lo, sel):
    return {'c': sub_lo + sel['symbols'], 'plus': STEPS[sel['plus']],
            'minus': STEPS[sel['minus']],
            'total_ok': list(RULES[sel['total_ok']]), 'min_ok': list(RULES[sel['min_ok']])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--positions', type=int, default=22)
    ap.add_argument('--sub-lo', type=int, required=True)
    ap.add_argument('--sub-hi', type=int, required=True)
    ap.add_argument('--rank', default='order')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, 'out', f'q1_c{a.sub_lo}-{a.sub_hi}.json')

    module, registry, frozen = build_module()
    program, signals = stage_b_dyck(module, registry, positions=a.positions,
                                    sub_range=range(a.sub_lo, a.sub_hi))
    s = prepare.load()
    tr = examples(s['train'])
    print(f'[{a.sub_lo},{a.sub_hi}) nodes {len(program.nodes)} space {space_size(program)}',
          flush=True)

    t0 = time.perf_counter()
    res = enumerate_fit(program, tr, signals, registry, tolerance=1e-6, rank=a.rank)
    wall = time.perf_counter() - t0

    rep = {'stream': common.STREAM_POST_AUDIT, 'scaffold': 'stage_b_dyck',
           'positions': a.positions, 'sub_range': [a.sub_lo, a.sub_hi],
           'nodes': len(program.nodes), 'candidate_counts_multi':
               {n.name: len(n.candidates) for n in program.nodes if len(n.candidates) > 1},
           'shard_space_size': space_size(program), 'full_space_size': 121 * 5 * 5 * 15 * 15,
           'seconds': wall, 'seconds_per_program': wall / max(1, res.evaluated),
           'train_episodes': len(s['train']),
           'enumeration': {k: v for k, v in res.to_dict().items() if k != 'selections'}}
    if res.solved:
        sel = res.selections
        rep['first_conforming'] = {
            'selections': sel, 'decoded': decode(program, a.sub_lo, sel),
            'global_enumeration_index': global_index(program, a.sub_lo, sel)}
        for name in ('train', 'heldout_seen_lengths', 'heldout_unseen_lengths'):
            acc, per_len = accuracy(program, sel, registry, s[name])
            rep[name] = {'accuracy': acc, 'per_length': per_len, **baselines(s[name])}
        chosen = program.harden(sel).pruned()
        rep['description_bits'] = chosen.description_bits(registry)
        rep['execution_cost'] = chosen.execution_cost(registry)
        rep['live_nodes'] = len(chosen.nodes)
    json.dump(rep, open(out, 'w'), indent=1)
    print('wrote', out, 'conforming', res.conforming, 'certificate', res.certificate,
          f'{wall:.0f}s', flush=True)


if __name__ == '__main__':
    main()
