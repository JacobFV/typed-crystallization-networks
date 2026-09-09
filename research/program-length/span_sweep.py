"""Q1, the part ranking cannot reach: shorten the *family*, and price the loss.

Enumeration certifies (see `visual_family.py`) that every conforming program in
each of the visual artifact's three declared spaces has the identical node count
and the identical bytecode count, so no cost term can shorten anything there.
The length is fixed by the scaffold, and in S2 by one scaffold parameter:
`rect_scaffold(..., span)` unrolls `span - 1` prefix-conjunction terms per axis,
nine nodes apiece.  `span` is a declared bound on a fixed-depth unrolling, not a
domain fact -- the same parameter would appear in any feed-forward formulation of
a bounded run length -- so sweeping it is the honest way to ask what a shorter
program in this family would cost.

At each span the space is exhausted, so "no conforming program at this length"
is a certificate rather than a search failure.  Task quality is reported on the
held-out episodes with its trivial baseline beside it, because a shorter program
that scores below a constant predictor has not bought anything.
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

from measure import compiled, dump, static                        # noqa: E402
from common import (FLAT, Registry, accuracy, all_conforming, episode,  # noqa: E402
                    exact_error)
import rung3_widgets as R                                          # noqa: E402
from tcn.search import program_cost, space_size                    # noqa: E402

TOL = 1e-6


def widget_extents(seeds, split, cfg):
    """The widths and heights the supervision actually contains, for context.

    Reported, not consulted: no span is chosen from it.  It is here so a reader
    can see why a span conforms or fails without re-deriving the episodes.
    """
    w = h = 0
    for s in seeds:
        ep = episode(s, split, **cfg)
        for d in ep['probes']['hierarchy']:
            if d['rect'][0] >= 1 and d['rect'][1] >= 1:
                w = max(w, d['rect'][2]); h = max(h, d['rect'][3])
    return {'max_width': w, 'max_height': h}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', type=int, default=6)
    ap.add_argument('--validation', type=int, default=3)
    ap.add_argument('--held', type=int, default=6)
    ap.add_argument('--spans', type=int, nargs='*', default=[4, 8, 12, 16, 20, 24, 28, 32])
    ap.add_argument('--tag', default='span_sweep')
    a = ap.parse_args()

    registry = Registry()
    cfg = dict(FLAT)
    probe = episode(0, 'train', **cfg)
    W, H = probe['width'], probe['height']
    offsets = R.offset_pool(W)
    shipped = json.loads((ROOT / 'research/visual-ladder/out/rung3.json').read_text())
    p0 = R.same_scaffold(registry, W, H)
    same_module = registry.register_module(p0.harden(shipped['s0']['chosen']))

    rtr = R.rect_examples(range(a.train), 'train', **cfg)
    rva = R.rect_examples(range(50, 50 + a.validation), 'validation', **cfg)
    rhe = R.rect_examples(range(100, 100 + a.held), 'test', **cfg)
    cases = [{k: row['inputs'][k].decoded for k, _ in
              R.rect_scaffold(registry, W, H, same_module, offsets, span=4).inputs}
             for row in rhe[:4]]

    result = {'width': W, 'height': H, 'spans': a.spans,
              'train_records': len(rtr), 'validation_records': len(rva),
              'held_records': len(rhe),
              'train_extents': widget_extents(range(a.train), 'train', cfg),
              'held_extents': widget_extents(range(100, 100 + a.held), 'test', cfg),
              'shipped_span': max(W, H), 'rows': []}
    print('extents', result['train_extents'], result['held_extents'])

    for span in a.spans:
        t0 = time.perf_counter()
        p2 = R.rect_scaffold(registry, W, H, same_module, offsets, span=span)
        walk = all_conforming(p2, rtr, R.rect_signals(), registry, TOL)
        conforming = walk.pop('conforming')
        survivors = [s for s in conforming
                     if exact_error(p2, rva, R.rect_signals(), registry, s) <= TOL]
        row = {'span': span, 'scaffold_nodes': len(p2.nodes), 'space_size': space_size(p2),
               'evaluated': walk['evaluated'], 'exhausted': walk['exhausted'],
               'conforming': len(conforming), 'survivors': len(survivors),
               'certificate': ('unique' if walk['exhausted'] and len(conforming) == 1 else
                               'none exists' if walk['exhausted'] and not conforming else
                               'complete' if walk['exhausted'] else 'budget-limited'),
               'seconds': time.perf_counter() - t0}
        if survivors:
            s = survivors[0]
            hardened = p2.harden(s)
            row.update(static(hardened, registry))
            bits, cost = program_cost(p2, s, registry)
            comp, _outs, _ = compiled(hardened, registry, cases)
            row.update({'ranked_description_bits': bits, 'ranked_execution_cost': cost,
                        'bytecodes_per_call': comp['bytecodes'][0],
                        'bytecodes_total': comp['bytecodes_total'],
                        'source_bytes': comp['source_bytes'],
                        'held_max_error': exact_error(p2, rhe, R.rect_signals(), registry, s),
                        'held_accuracy': accuracy(p2, rhe, R.rect_signals(), registry, s),
                        'train_accuracy': accuracy(p2, rtr, R.rect_signals(), registry, s),
                        'steps_chosen': [offsets[s['step_w']], offsets[s['step_h']]]})
        result['rows'].append(row)
        print(f"span {span:3d}  nodes {row['scaffold_nodes']:4d}  conforming {row['conforming']}"
              f"  certificate {row['certificate']:12s}"
              f"  bytecodes {row.get('bytecodes_per_call', '-')}"
              f"  held acc {row.get('held_accuracy', '-')}")
        dump(a.tag, result)


if __name__ == '__main__':
    main()
