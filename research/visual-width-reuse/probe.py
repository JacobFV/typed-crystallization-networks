"""Feasibility probe: what the gui generator actually achieves per resolution.

Reported, not consulted for any certificate.  Its only job is to fix the width
grid before the pre-registration is written: which resolutions the generator can
serve at all, how many non-root widget corners each supplies (the S2 supervision),
and the achieved widget extents -- because `rect_scaffold`'s `span` must exceed
the longest extent or no program in the family can conform (§41).
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(ROOT / 'research' / 'visual-ladder')):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import FLAT, episode          # noqa: E402
import rung3_widgets as R                 # noqa: E402

RESOLUTIONS = (12, 16, 20, 24, 32, 40, 48)


def main():
    rows = []
    for res in RESOLUTIONS:
        cfg = dict(FLAT)
        cfg['resolution'] = res
        t0 = time.perf_counter()
        try:
            eps = [episode(s, 'train', **cfg) for s in range(3)]
        except Exception as exc:                       # noqa: BLE001
            rows.append({'resolution': res, 'error': f'{type(exc).__name__}: {exc}'})
            print(res, 'FAILED', type(exc).__name__, exc)
            continue
        corners = 0
        max_w = max_h = 0
        for ep in eps:
            for d in ep['probes']['hierarchy']:
                if d['rect'][0] >= 1 and d['rect'][1] >= 1:
                    corners += 1
                    max_w = max(max_w, d['rect'][2])
                    max_h = max(max_h, d['rect'][3])
        W, H = eps[0]['width'], eps[0]['height']
        row = {'resolution': res, 'width': W, 'height': H,
               'observation_bytes': 3 * W * H,
               'achieved_widgets': [len(e['probes']['hierarchy']) for e in eps],
               'non_root_corners_3eps': corners,
               'max_widget_width': max_w, 'max_widget_height': max_h,
               'span_default': max(W, H), 'offset_pool': list(R.offset_pool(W)),
               'interior_positions': (W - 1) * (H - 1),
               'generation_seconds_3eps': time.perf_counter() - t0}
        rows.append(row)
        print(json.dumps(row))
    (HERE / 'out' / 'probe.json').write_text(json.dumps({'rows': rows}, indent=1))


if __name__ == '__main__':
    main()
