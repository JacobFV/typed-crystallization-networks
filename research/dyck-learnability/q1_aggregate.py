"""Q1: union the eleven shards into one statement about the full 680,625-program space.

A union of exhausted, disjoint shards covering `c in [0,121)` is an exhaustive
decision of the whole space, and the certificate is `complete` only if **every**
shard is `exhausted` and complete. If any shard is not, the union carries **no**
certificate and is reported as a bounded search, which supports neither "no
solution exists" nor "search failed".

Because the `c` partition is order-preserving in `enumerate_fit`'s mixed-radix
walk (see `q1_enum.py`), the global first conforming member is the first
conforming member of the lowest-`c` shard that has one. The index arithmetic is
checked here against section 45's own recorded witness index, 571,822.
"""
from __future__ import annotations
import sys, os, json, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

FULL = 121 * 5 * 5 * 15 * 15
WITNESS = {'c': 101, 'plus': 1, 'minus': -1, 'total_ok': ('eq', 0), 'min_ok': ('ge', 0)}
STEPS = (-2, -1, 0, 1, 2)
RULES = [(op, v) for v in (-2, -1, 0, 1, 2) for op in ('eq', 'ge', 'le')]


def witness_index():
    return ((((WITNESS['c'] * 5 + STEPS.index(WITNESS['plus'])) * 5
              + STEPS.index(WITNESS['minus'])) * 15 + RULES.index(WITNESS['total_ok'])) * 15
            + RULES.index(WITNESS['min_ok']))


def main():
    shards = []
    for path in sorted(glob.glob(os.path.join(HERE, 'out', 'q1_c*.json')),
                       key=lambda p: int(os.path.basename(p)[4:].split('-')[0])):
        shards.append(json.load(open(path)))
    lo_hi = [tuple(s['sub_range']) for s in shards]
    covered = set()
    for lo, hi in lo_hi:
        covered |= set(range(lo, hi))
    overlaps = sum(hi - lo for lo, hi in lo_hi) - len(covered)

    evaluated = sum(s['enumeration']['evaluated'] for s in shards)
    space = sum(s['shard_space_size'] for s in shards)
    conforming = sum(s['enumeration']['conforming'] for s in shards)
    all_exhausted = all(s['enumeration']['exhausted'] for s in shards)
    all_complete = all(s['enumeration']['certificate'] == 'complete' for s in shards)
    wall = max(s['seconds'] for s in shards)
    cpu = sum(s['seconds'] for s in shards)
    firsts = [s for s in shards if 'first_conforming' in s]
    firsts.sort(key=lambda s: s['first_conforming']['global_enumeration_index'])

    union_cert = ('complete' if (all_exhausted and all_complete and covered == set(range(0, 121))
                                 and overlaps == 0 and space == FULL)
                  else 'none')
    if union_cert == 'complete' and conforming == 1:
        union_cert = 'unique'

    rep = {'question': 'Q1 -- does the unmodified search find the min-prefix program '
                       'without the hand-chosen window?',
           'scaffold': 'stage_b_dyck', 'positions': 22,
           'full_space_size': FULL, 'shards': len(shards),
           'c_values_covered': len(covered), 'c_overlaps': overlaps,
           'partition_is_exact': covered == set(range(0, 121)) and overlaps == 0,
           'summed_shard_space': space, 'evaluated': evaluated,
           'exhausted': all_exhausted and space == FULL and evaluated == FULL,
           'conforming': conforming, 'certificate': union_cert,
           'wall_clock_seconds_parallel': wall,
           'cpu_seconds_total': cpu, 'cpu_hours_total': cpu / 3600,
           'seconds_per_program': cpu / max(1, evaluated),
           'witness_index_recomputed': witness_index(),
           'witness_index_recorded_by_section_45': 571822,
           'index_arithmetic_agrees': witness_index() == 571822,
           'per_shard': [{'sub_range': s['sub_range'], 'space': s['shard_space_size'],
                          'evaluated': s['enumeration']['evaluated'],
                          'exhausted': s['enumeration']['exhausted'],
                          'conforming': s['enumeration']['conforming'],
                          'certificate': s['enumeration']['certificate'],
                          'seconds': s['seconds'],
                          'first_conforming_index':
                              s.get('first_conforming', {}).get('global_enumeration_index')}
                         for s in shards]}
    if firsts:
        f = firsts[0]
        rep['first_conforming'] = {
            'decoded': f['first_conforming']['decoded'],
            'global_enumeration_index': f['first_conforming']['global_enumeration_index'],
            'fraction_through_enumeration': f['first_conforming']['global_enumeration_index'] / FULL,
            'programs_before_it': f['first_conforming']['global_enumeration_index'],
            'projected_single_thread_hours_to_reach':
                f['first_conforming']['global_enumeration_index'] * (cpu / max(1, evaluated)) / 3600,
            'train': f['train'], 'heldout_seen_lengths': f['heldout_seen_lengths'],
            'heldout_unseen_lengths': f['heldout_unseen_lengths'],
            'description_bits': f.get('description_bits'),
            'execution_cost': f.get('execution_cost')}
    ctrl_path = os.path.join(HERE, 'out', 'q1b_c95-110.json')
    if os.path.exists(ctrl_path):
        c = json.load(open(ctrl_path))
        rep['arm_1b_control_section_45_window'] = {
            'sub_range': c['sub_range'], 'space_size': c['shard_space_size'],
            'evaluated': c['enumeration']['evaluated'],
            'exhausted': c['enumeration']['exhausted'],
            'conforming': c['enumeration']['conforming'],
            'certificate': c['enumeration']['certificate'], 'seconds': c['seconds'],
            'recorded_by_section_45': {'space_size': 84375, 'evaluated': 84375,
                                       'exhausted': True, 'conforming': 110,
                                       'certificate': 'complete'},
            'reproduces': (c['shard_space_size'] == 84375 and c['enumeration']['evaluated'] == 84375
                           and c['enumeration']['exhausted'] and c['enumeration']['conforming'] == 110
                           and c['enumeration']['certificate'] == 'complete'),
            'heldout_unseen_lengths': c.get('heldout_unseen_lengths')}
    out = os.path.join(HERE, 'out', 'q1_union.json')
    json.dump(rep, open(out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != 'per_shard'}, indent=1))
    print('\nper shard:')
    for r in rep['per_shard']:
        print(' ', r)
    print('wrote', out)


if __name__ == '__main__':
    main()
