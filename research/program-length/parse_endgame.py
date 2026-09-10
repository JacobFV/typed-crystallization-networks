"""The whole parse, before and after both levers, against the hand-written arm D.

`research/compiled-runtime/RESULTS.md` section 7 measured the compiled parse at
650,567 bytecodes against the hand-written reference's 58,862 -- 11.1x -- and
attributed the remaining 2.25x to per-bytecode cost.  This script reprices that
single number under every shortening this track found:

  shipped   the artifact as frozen, span 32
  shortest  the shortest conforming rect module the enumeration certifies (the
            `--span` argument), assembled into the same parse
  arm D     `research/compiled-runtime/fixtures.py`'s hand-written parse

The rectangle set is asserted identical across all three on every held-out screen
before any count is reported; `frozenset == frozenset` on the six-tuples, which
is the same assertion `bench.py` makes.
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(ROOT / 'research' / 'visual-ladder'), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from measure import count_opcodes, dump, static                   # noqa: E402
from common import FLAT, Registry, bytes_type, episode            # noqa: E402
import rung3_widgets as R                                          # noqa: E402
from tcn.compile import compile_program                            # noqa: E402


def timed(fn, reps):
    gc.disable()
    try:
        s = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            s.append((time.perf_counter() - t0) * 1e3)
    finally:
        gc.enable()
    return {'median_ms': statistics.median(s), 'min_ms': min(s)}


def reference(c):
    """Arm D, copied verbatim from `research/compiled-runtime/fixtures.py`."""
    px, W, H = c['observation'], c['_w'], c['_h']
    out = set()
    for y in range(1, H):
        row = y * W
        for x in range(1, W):
            i = 3 * (row + x)
            col = (px[i], px[i + 1], px[i + 2])
            if col == (px[i - 3], px[i - 2], px[i - 1]):
                continue
            u = i - 3 * W
            if col == (px[u], px[u + 1], px[u + 2]):
                continue
            w = 1
            while x + w < W and (px[i + 3 * w], px[i + 3 * w + 1], px[i + 3 * w + 2]) == col:
                w += 1
            h = 1
            while y + h < H and (px[i + 3 * W * h], px[i + 3 * W * h + 1], px[i + 3 * W * h + 2]) == col:
                h += 1
            out.add((x, y, w, h,
                     col[0] | col[1] << 8 | col[2] << 16,
                     px[i - 3] | px[i - 2] << 8 | px[i - 1] << 16))
    return {'mapped': frozenset(out)}


def build_parse(span, cfg, seeds):
    """The S3 assembly over an S2 rect module unrolled to `span`."""
    registry = Registry()
    probe = episode(0, 'train', **cfg)
    W, H = probe['width'], probe['height']
    offsets = R.offset_pool(W)
    shipped = json.loads((ROOT / 'research/visual-ladder/out/rung3.json').read_text())
    p0 = R.same_scaffold(registry, W, H)
    m0 = registry.register_module(p0.harden(shipped['s0']['chosen']))
    p1 = R.corner_scaffold(registry, W, H, m0, offsets)
    m1 = registry.register_module(p1.harden(shipped['s1']['chosen']))
    p2 = R.rect_scaffold(registry, W, H, m0, offsets, span=span)
    sel = dict(shipped['s2']['chosen'])
    sel = {n.name: sel.get(n.name, 0) for n in p2.nodes}
    m2 = registry.register_module(p2.harden(sel))
    eps = [episode(s, 'test', **cfg) for s in seeds]
    BT = bytes_type(eps[0]['width'], eps[0]['height'])
    prog = R.assembly(registry, BT, R.interior_positions(eps[0]), m1, m2)
    cases = [{'observation': tuple(e['pixels']), '_w': e['width'], '_h': e['height']}
             for e in eps]
    return prog, registry, cases, static(p2.harden(sel), registry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spans', type=int, nargs='*', default=[32])
    ap.add_argument('--seeds', type=int, nargs='*', default=[200, 201, 202])
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--tag', default='parse_endgame')
    a = ap.parse_args()
    cfg = dict(FLAT)

    base_cases = build_parse(32, cfg, a.seeds)[2]
    want = [reference(c) for c in base_cases]
    out = {'seeds': a.seeds, 'rows': [],
           'arm_d_bytecodes': [count_opcodes(lambda c=c: reference(c)) for c in base_cases],
           'arm_d_time': timed(lambda: reference(base_cases[0]), a.reps * 4)}

    for span in a.spans:
        prog, registry, cases, rect_static = build_parse(span, cfg, a.seeds)
        res = compile_program(prog, registry)
        mod = res.module()
        got = [mod.run({'observation': c['observation']}, validate=False)[0] for c in cases]
        identical = all(g['mapped'] == w['mapped'] for g, w in zip(got, want))
        row = {'span': span, 'rect_module': rect_static, 'identical_to_arm_d': identical,
               'source_bytes': len(res.source), 'stats': dict(res.stats),
               'rects': [len(g['mapped']) for g in got]}
        if identical:
            row['bytecodes'] = [count_opcodes(
                lambda c=c: mod.run({'observation': c['observation']}, validate=False))
                for c in cases]
            row['time_program_only'] = timed(
                lambda: mod.run({'observation': cases[0]['observation']}, validate=False), a.reps)
            row['time_complete'] = timed(
                lambda: mod.run({'observation': cases[0]['observation']}, validate=True), a.reps)
            row['bytecodes_over_arm_d'] = row['bytecodes'][0] / out['arm_d_bytecodes'][0]
        out['rows'].append(row)
        print(span, {k: v for k, v in row.items() if k not in ('stats', 'rect_module')})
        dump(a.tag, out)
    print('arm D bytecodes', out['arm_d_bytecodes'])
    dump(a.tag, out)


if __name__ == '__main__':
    main()
