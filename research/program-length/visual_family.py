"""Q1 on the visual artifact: enumerate the family, then ask what ranking buys.

The visual parse is the artifact whose compiled program is 27.6x hand-written
Python, decomposed by `research/compiled-runtime/RESULTS.md` section 7 into
11.1x more bytecodes times 2.25x per-bytecode cost.  The brief's hypothesis is
that a cost term in the search would attack the 11.1x.

This script does not assume the hypothesis is testable by tuning a weight.  It
enumerates the *whole declared space* of each of the three searched stages,
collects every conforming program, and reports each one's four size numbers
(ARCHITECTURE section 8.1) including the bytecodes its compiled form actually
executes.  Ranking by description bits or by execution cost can only ever pick
among these; where every member has the same numbers, ranking is inert and the
enumeration certificate says so, which is a different statement from "the search
failed to find a shorter program".

Runs the three stages at the same configuration `rung3_widgets.main()` used, so
the "before" column is the shipped artifact rather than a re-derivation.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(ROOT / 'research' / 'visual-ladder'), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from measure import compiled, dump, static                      # noqa: E402
from common import (FLAT, Registry, accuracy, all_conforming, bytes_type,   # noqa: E402
                    episode, exact_error, record_type)
import rung3_widgets as R                                        # noqa: E402
from tcn.search import program_cost, space_size                  # noqa: E402
from tcn.types import Value                                      # noqa: E402

TOL = 1e-6


def _native(program, rows, limit):
    """Decoded input dicts for the compiled arm, from the same example rows."""
    keys = [k for k, _ in program.inputs]
    return [{k: row['inputs'][k].decoded for k in keys} for row in rows[:limit]]


def survey(name, program, registry, train, validation, held, signals, cases,
           chosen_from_artifact):
    """Enumerate, measure every conforming program, and report what ranking picks."""
    t0 = time.perf_counter()
    walk = all_conforming(program, train, signals, registry, TOL)
    conforming = walk.pop('conforming')
    survivors = [s for s in conforming
                 if exact_error(program, validation, signals, registry, s) <= TOL]

    rows = []
    for s in survivors:
        hardened = program.harden(s)
        st = static(hardened, registry)
        bits, cost = program_cost(program, s, registry)
        comp, outs, _ = compiled(hardened, registry, cases)
        rows.append({'selections': s, **st, 'ranked_description_bits': bits,
                     'ranked_execution_cost': cost,
                     'bytecodes': comp['bytecodes'], 'bytecodes_total': comp['bytecodes_total'],
                     'source_bytes': comp['source_bytes'],
                     'held_max_error': exact_error(program, held, signals, registry, s),
                     'held_accuracy': accuracy(program, held, signals, registry, s),
                     'outputs': [json.dumps(o, sort_keys=True, default=str) for o in outs]})

    def pick(key):
        return min(range(len(rows)), key=lambda i: key(rows[i])) if rows else None

    idx = {'order': 0 if rows else None,
           'description': pick(lambda r: (r['ranked_description_bits'], r['ranked_execution_cost'])),
           'cost': pick(lambda r: (r['ranked_execution_cost'], r['ranked_description_bits'])),
           'bytecodes': pick(lambda r: r['bytecodes_total'])}

    out = {'stage': name, 'space_size': space_size(program),
           'scaffold_nodes': len(program.nodes),
           'enumeration': walk, 'conforming': len(conforming), 'survivors': len(survivors),
           'certificate': 'exhausted' if walk['exhausted'] else 'budget-limited',
           'size_degenerate': (len({r['nodes'] for r in rows}) <= 1 and
                               len({r['bytecodes_total'] for r in rows}) <= 1),
           'distinct_nodes': sorted({r['nodes'] for r in rows}),
           'distinct_description_bits': sorted({r['ranked_description_bits'] for r in rows}),
           'distinct_execution_cost': sorted({r['ranked_execution_cost'] for r in rows}),
           'distinct_bytecodes': sorted({r['bytecodes_total'] for r in rows}),
           'rank_picks': idx,
           'chosen_matches_artifact': bool(rows) and rows[0]['selections'] == chosen_from_artifact,
           'outputs_identical_across_survivors': (
               len({tuple(r['outputs']) for r in rows}) <= 1),
           'rows': rows, 'seconds': time.perf_counter() - t0}
    print(f"[{name}] space {out['space_size']} conforming {out['conforming']} "
          f"survivors {out['survivors']} certificate {out['certificate']}")
    print(f"[{name}] nodes {out['distinct_nodes']} bits {out['distinct_description_bits']} "
          f"cost {out['distinct_execution_cost']} bytecodes {out['distinct_bytecodes']}")
    print(f"[{name}] ranking picks {idx}; size-degenerate={out['size_degenerate']}")
    return out, survivors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', type=int, default=6)
    ap.add_argument('--validation', type=int, default=3)
    ap.add_argument('--held', type=int, default=6)
    ap.add_argument('--pairs', type=int, default=48)
    ap.add_argument('--corner-per-image', type=int, default=120)
    ap.add_argument('--stages', default='s0,s1,s2')
    ap.add_argument('--tag', default='visual_family')
    a = ap.parse_args()
    want = set(a.stages.split(','))

    registry = Registry()
    cfg = dict(FLAT)
    probe = episode(0, 'train', **cfg)
    W, H = probe['width'], probe['height']
    offsets = R.offset_pool(W)
    shipped = json.loads((ROOT / 'research/visual-ladder/out/rung3.json').read_text())
    result = {'configuration': cfg, 'width': W, 'height': H, 'offsets': list(offsets),
              'arguments': vars(a)}

    # ---- S0 -------------------------------------------------------------
    tr = R.same_examples(range(a.train), 'train', a.pairs, seed=1, **cfg)
    va = R.same_examples(range(50, 50 + a.validation), 'validation', a.pairs, seed=2, **cfg)
    he = R.same_examples(range(100, 100 + a.held), 'test', a.pairs, seed=3, **cfg)
    p0 = R.same_scaffold(registry, W, H)
    if 's0' in want:
        result['s0'], surv0 = survey('S0 same', p0, registry, tr, va, he, R.same_signals(),
                                     _native(p0, he, 8), shipped['s0']['chosen'])
    same_module = registry.register_module(p0.harden(shipped['s0']['chosen']))

    # ---- S1 -------------------------------------------------------------
    p1 = R.corner_scaffold(registry, W, H, same_module, offsets)
    if 's1' in want:
        ctr = R.corner_examples(range(a.train), 'train', a.corner_per_image, seed=4, **cfg)
        cva = R.corner_examples(range(50, 50 + a.validation), 'validation', a.corner_per_image,
                                seed=5, **cfg)
        che = R.corner_examples(range(100, 100 + a.held), 'test', a.corner_per_image, seed=6, **cfg)
        result['s1'], _ = survey('S1 corner', p1, registry, ctr, cva, che, R.corner_signals(),
                                 _native(p1, che, 8), shipped['s1']['chosen'])
    corner_module = registry.register_module(p1.harden(shipped['s1']['chosen']))

    # ---- S2 -------------------------------------------------------------
    p2 = R.rect_scaffold(registry, W, H, same_module, offsets)
    if 's2' in want:
        rtr = R.rect_examples(range(a.train), 'train', **cfg)
        rva = R.rect_examples(range(50, 50 + a.validation), 'validation', **cfg)
        rhe = R.rect_examples(range(100, 100 + a.held), 'test', **cfg)
        result['s2'], _ = survey('S2 rect', p2, registry, rtr, rva, rhe, R.rect_signals(),
                                 _native(p2, rhe, 4), shipped['s2']['chosen'])

    dump(a.tag, result)


if __name__ == '__main__':
    main()
