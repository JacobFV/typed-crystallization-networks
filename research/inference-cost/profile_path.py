"""Where the milliseconds inside `program.run` actually go.

`cProfile` over one screenshot parse, with the per-function `tottime` grouped
into four roles. The absolute seconds are inflated by the profiler (about 3.5x);
the shares are the point, and they are what makes the difference between "the
interpreter is inherently slow" and "the value layer re-encodes at every edge".
"""
import sys, cProfile, pstats, json
ROOT = '/home/brandonin/Documents/typed-crystallization-networks'
for p in (ROOT, ROOT + '/research/visual-ladder', ROOT + '/research/inference-cost'):
    sys.path.insert(0, p)
from harness import dump
from common import FLAT, Registry, bytes_type, episode, load
from tcn.types import Value
import rung3_widgets as R

# Keyed by (file fragment, line number) where a line number disambiguates a
# generator expression from the function that owns it: `types.py:175` is inside
# `decode` and `types.py:142` is inside `encode`, and grouping them by name alone
# charges both to whichever role is tested first.
ROLES = {
    'Type.decode (types.py:171,175)': [('tcn/types.py', 171), ('tcn/types.py', 175)],
    'Type.encode (types.py:134,142)': [('tcn/types.py', 134), ('tcn/types.py', 142)],
    'validate_raw (types.py:185)': [('tcn/types.py', 185)],
    'numeric/type guards called by both': [(None, 'isfinite'), (None, 'isinstance'),
                                           (None, 'getattr'), (None, 'struct.unpack'),
                                           (None, 'struct.pack')],
    'Type/Value dict round-trip': [('tcn/types.py', 81), ('tcn/types.py', 89),
                                   ('dataclasses.py', None)],
    'operator semantics + graph walk': [('tcn/operators.py', None), ('tcn/graph.py', None)],
}


def main():
    found = load('rung3'); registry = Registry()
    probe = episode(0, 'train', **FLAT); W, H = probe['width'], probe['height']
    offsets = R.offset_pool(W)
    p0 = R.same_scaffold(registry, W, H); m0 = registry.register_module(p0.harden(found['s0']['chosen']))
    p1 = R.corner_scaffold(registry, W, H, m0, offsets); m1 = registry.register_module(p1.harden(found['s1']['chosen']))
    p2 = R.rect_scaffold(registry, W, H, m0, offsets); m2 = registry.register_module(p2.harden(found['s2']['chosen']))
    ep = episode(200, 'test', **FLAT); BT = bytes_type(ep['width'], ep['height'])
    parser = R.assembly(registry, BT, R.interior_positions(ep), m1, m2)
    v = Value.of(BT, ep['pixels'])
    pr = cProfile.Profile(); pr.enable()
    parser.run({'observation': v}, registry=registry)
    pr.disable()
    stats = pstats.Stats(pr)
    total = stats.total_tt
    buckets = {k: 0.0 for k in ROLES} | {'other': 0.0}
    lines = []
    for (fn, lineno, name), (cc, nc, tt, ct, _) in stats.stats.items():
        placed = None
        for role, patterns in ROLES.items():
            for f, n in patterns:
                if f is not None and f not in fn:
                    continue
                if isinstance(n, int):
                    if lineno == n: placed = role
                elif n is None or n in name:
                    placed = role
                if placed: break
            if placed: break
        buckets[placed or 'other'] += tt
        if tt > 0.05:
            lines.append({'function': f'{fn.split("/")[-1]}:{lineno}({name})', 'calls': nc,
                          'tottime_s': tt, 'role': placed or 'other'})
    out = {'profiled_total_seconds': total,
           'note': 'cProfile inflates wall clock ~3.5x; the shares are the measurement',
           'by_role_seconds': buckets,
           'by_role_share': {k: v / total for k, v in buckets.items()},
           'hot_functions': sorted(lines, key=lambda r: -r['tottime_s'])}
    print(json.dumps({k: out[k] for k in ('profiled_total_seconds', 'by_role_share')}, indent=1))
    dump('profile', out)


if __name__ == '__main__':
    main()
