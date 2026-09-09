"""Complete inference path for the screenshot-to-hierarchy parse.

The artifact is `research/visual-ladder`'s S3 assembly, rebuilt from the frozen
selections in `out/rung3.json` exactly as `parse_report.py` rebuilds it. Nothing
is searched, fitted or tuned here; this file only measures.

Path stages, in the order a deployment would run them:
  acquire  the generator renders a screen to raw RGB bytes (stands in for a
           screenshot grab; a real one would be a framebuffer read)
  encode   `Value.of(bytes_type, pixels)` -- the typed carrier
  execute  `program.run` -- the four caller nodes over 961 interior positions
  decode   `.decoded` on the output set -- back to plain tuples

Two references are measured on the identical input and checked to produce the
identical output set:
  python   the same rule hand-written as a plain nested loop
  fixed    the same TCN program with two mechanical caching changes monkeypatched
           into `tcn.types.Value` (memoized `decoded`, no re-validation of
           carriers the interpreter itself produced). This separates "interpreter
           overhead that is a bug" from "interpreter overhead that is inherent".
"""
import sys, json, time
sys.path.insert(0, '/home/brandonin/Documents/typed-crystallization-networks')
sys.path.insert(0, '/home/brandonin/Documents/typed-crystallization-networks/research/visual-ladder')
sys.path.insert(0, '/home/brandonin/Documents/typed-crystallization-networks/research/inference-cost')

from harness import Counter, timed, dump, rss_mb, size_report
from common import FLAT, Registry, bytes_type, episode, load
import tcn.types as TT
from tcn.types import Value
import rung3_widgets as R

SEEDS = [200, 201, 202]


def build():
    found = load('rung3')
    registry = Registry()
    probe = episode(0, 'train', **FLAT)
    W, H = probe['width'], probe['height']
    offsets = R.offset_pool(W)
    p0 = R.same_scaffold(registry, W, H); m0 = registry.register_module(p0.harden(found['s0']['chosen']))
    p1 = R.corner_scaffold(registry, W, H, m0, offsets); m1 = registry.register_module(p1.harden(found['s1']['chosen']))
    p2 = R.rect_scaffold(registry, W, H, m0, offsets); m2 = registry.register_module(p2.harden(found['s2']['chosen']))
    return registry, m1, m2, W, H


def plain_python(px, W, H):
    """The same parse, written directly. `px` is the flat RGB byte tuple.

    Corner  : pixel differs from its left AND its upper neighbour (S1, truth_1).
    Extent  : length of the contiguous same-colour run right / down (S2).
    Keys    : the pixel's own packed RGB, and its left neighbour's (S2).
    """
    out = set()
    for y in range(1, H):
        row = y * W
        for x in range(1, W):
            i = 3 * (row + x)
            c = (px[i], px[i + 1], px[i + 2])
            if c == (px[i - 3], px[i - 2], px[i - 1]):
                continue
            u = i - 3 * W
            if c == (px[u], px[u + 1], px[u + 2]):
                continue
            w = 1
            while x + w < W and (px[i + 3 * w], px[i + 3 * w + 1], px[i + 3 * w + 2]) == c:
                w += 1
            h = 1
            while y + h < H and (px[i + 3 * W * h], px[i + 3 * W * h + 1], px[i + 3 * W * h + 2]) == c:
                h += 1
            out.add((x, y, w, h,
                     c[0] | c[1] << 8 | c[2] << 16,
                     px[i - 3] | px[i - 2] << 8 | px[i - 1] << 16))
    return out


def main():
    t = time.perf_counter()
    registry, m1, m2, W, H = build()
    build_s = time.perf_counter() - t

    rows = []
    for seed in SEEDS:
        acquire_ms, _, ep = timed(lambda: episode(seed, 'test', **FLAT))
        BT = bytes_type(ep['width'], ep['height'])
        positions = R.interior_positions(ep)
        parser = R.assembly(registry, BT, positions, m1, m2)
        encode_ms, _, value = timed(lambda: Value.of(BT, ep['pixels']))
        with Counter(registry) as counter:
            execute_ms, execute_min, got = timed(lambda: parser.run({'observation': value}, registry=registry)[0])
        ops = counter.n // 3
        decode_ms, _, decoded = timed(lambda: set(got['mapped'].decoded))
        py_ms, py_min, py_out = timed(lambda: plain_python(ep['pixels'], ep['width'], ep['height']), repeats=11)
        rows.append({'seed': seed, 'positions': len(positions),
                     'acquire_ms': acquire_ms, 'encode_ms': encode_ms,
                     'execute_ms': execute_ms, 'execute_min_ms': execute_min,
                     'decode_ms': decode_ms,
                     'total_ms': acquire_ms + encode_ms + execute_ms + decode_ms,
                     'operator_applications': ops,
                     'us_per_operator': execute_ms * 1000 / max(1, ops),
                     'plain_python_ms': py_ms, 'plain_python_min_ms': py_min,
                     'rects': len(decoded), 'python_rects': len(py_out),
                     'outputs_identical': decoded == py_out})
        print(json.dumps(rows[-1], indent=1))

    # --- the same program with the two caching changes, outputs checked identical ---
    def decoded_cached(self):
        try: return object.__getattribute__(self, '_dec')
        except AttributeError: pass
        d = TT.decode(self.type, self.raw); object.__setattr__(self, '_dec', d); return d
    ep = episode(SEEDS[0], 'test', **FLAT)
    BT = bytes_type(ep['width'], ep['height'])
    parser = R.assembly(registry, BT, R.interior_positions(ep), m1, m2)
    reference = set(parser.run({'observation': Value.of(BT, ep['pixels'])}, registry=registry)[0]['mapped'].decoded)
    Value.decoded = property(decoded_cached)
    memo_ms, _, a = timed(lambda: set(parser.run({'observation': Value.of(BT, ep['pixels'])},
                                                 registry=registry)[0]['mapped'].decoded), repeats=3)
    Value.__post_init__ = lambda self: None
    both_ms, _, b = timed(lambda: set(parser.run({'observation': Value.of(BT, ep['pixels'])},
                                                 registry=registry)[0]['mapped'].decoded), repeats=3)
    fixed = {'memoized_decode_ms': memo_ms, 'memoized_plus_no_revalidate_ms': both_ms,
             'outputs_identical': bool(a == reference and b == reference)}
    print(json.dumps(fixed, indent=1))

    report = {'artifact': 'visual-ladder S3 screenshot->hierarchy parse',
              'screen': {'width': W, 'height': H, 'bytes': 3 * W * H, 'configuration': FLAT},
              'module_build_seconds': build_s,
              'size': size_report(parser, registry),
              'episodes': rows, 'fixable': fixed, 'peak_rss_mb': rss_mb()}
    dump('parse', report)


if __name__ == '__main__':
    main()
