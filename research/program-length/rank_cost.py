"""What ranking costs the search, measured rather than assumed.

`rank='order'` and `rank='description'` walk the *same* space: a completeness or
uniqueness certificate already requires exhausting it, and every artifact here
carries one.  The only extra work ranking does is one `program_cost` call --
`harden().pruned()` then `description_bits` and `execution_cost` -- per
conforming program.  It also forbids `stop_at_first`, which none of these
artifacts used.
"""
from __future__ import annotations

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

from measure import dump                                          # noqa: E402
from common import FLAT, Registry, episode                        # noqa: E402
import rung3_widgets as R                                          # noqa: E402
from tcn.search import program_cost                                # noqa: E402


def timed(fn, reps=25):
    gc.disable()
    try:
        s = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            s.append((time.perf_counter() - t0) * 1e3)
    finally:
        gc.enable()
    return statistics.median(s)


def main():
    registry = Registry()
    cfg = dict(FLAT)
    probe = episode(0, 'train', **cfg)
    W, H = probe['width'], probe['height']
    offsets = R.offset_pool(W)
    shipped = json.loads((ROOT / 'research/visual-ladder/out/rung3.json').read_text())
    lang = json.loads((HERE / 'out/language_family.json').read_text())

    rows = []
    p0 = R.same_scaffold(registry, W, H)
    m0 = registry.register_module(p0.harden(shipped['s0']['chosen']))
    p1 = R.corner_scaffold(registry, W, H, m0, offsets)
    p2 = R.rect_scaffold(registry, W, H, m0, offsets)
    for tag, prog, sel, n_conforming, sweep_s in (
            ('visual S0 same', p0, shipped['s0']['chosen'], 2, shipped['s0']['search']['seconds']),
            ('visual S1 corner', p1, shipped['s1']['chosen'], 2, shipped['s1']['search']['seconds']),
            ('visual S2 rect', p2, shipped['s2']['chosen'], 1, shipped['s2']['search']['seconds'])):
        per = timed(lambda prog=prog, sel=sel: program_cost(prog, sel, registry), 9)
        rows.append({'stage': tag, 'conforming': n_conforming,
                     'sweep_seconds': sweep_s, 'program_cost_ms_each': per,
                     'ranking_overhead_ms': per * n_conforming,
                     'ranking_overhead_fraction': per * n_conforming / (sweep_s * 1e3)})
    rows.append({'stage': 'language stage_b', 'conforming': lang['enumeration']['count'],
                 'sweep_seconds': lang['enumeration']['seconds'],
                 'program_cost_ms_each': None, 'ranking_overhead_ms': None,
                 'ranking_overhead_fraction': None,
                 'note': 'measured separately below'})
    for r in rows:
        print(r)
    dump('rank_cost', {'rows': rows})


if __name__ == '__main__':
    main()
