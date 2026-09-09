"""The three frozen artifacts, their inputs, and the hand-written reference.

Nothing is searched, fitted or re-selected here.  Each fixture rebuilds the
program its own track froze, exactly as `research/inference-cost` rebuilds it,
and returns:

    program    the frozen `Program`
    registry   the `Registry` it was frozen against
    cases      a list of native (already decoded) input dicts
    reference  the hand-written plain-Python function on the same native input
    compare    how the reference's output is compared with the program's

`reference` is the arm-D program.  It is the same function written directly, and
it is checked against arm A before any timing is reported.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Two tracks ship a module called `common`; put the wanted one first and drop
# any stale import, so one process can build either fixture.
_TRACK_MODULES = ('common', 'scaffolds', 'rung3_widgets', 'run_stage_b', 'program', 'task')


def _use(track):
    d = str(ROOT / 'research' / track)
    for m in _TRACK_MODULES:
        sys.modules.pop(m, None)
    while d in sys.path:
        sys.path.remove(d)
    sys.path.insert(0, d)


# ---------------------------------------------------------------------------
# mixed -- the historical reference: answer = sin(xor(a, b) + x), 4 operations
# ---------------------------------------------------------------------------
def mixed():
    from tcn.runtime import load_program
    program, registry = load_program(ROOT / 'artifacts/demo/synthesis/program.json')
    cases = [{'a': True, 'b': False, 'x': 0.13},
             {'a': True, 'b': False, 'x': -0.43},
             {'a': False, 'b': True, 'x': 0.3},
             {'a': True, 'b': True, 'x': -0.2}]

    def reference(c):
        """`sin(xor(a, b) + x)`, written directly."""
        return {'answer': math.sin(float(c['a'] != c['b']) + c['x'])}

    return {'name': 'mixed', 'program': program, 'registry': registry, 'cases': cases,
            'reference': reference, 'tolerance': 0.0,
            'what': 'answer = sin(xor(a, b) + x) -- the historical 4-operation reference'}


# ---------------------------------------------------------------------------
# language -- grammaticality of a Dyck word from raw prompt bytes, 164 operations
# ---------------------------------------------------------------------------
def language():
    _use('language-capability')
    import common as LC
    import scaffolds
    from run_stage_b import build_module

    sel = json.load(open(ROOT / 'research/language-capability/stage_b.json'))['enumeration']['selections']
    module, registry, _ = build_module()
    prog, _ = scaffolds.stage_b(module, registry)
    program = prog.harden(sel).pruned()
    # Lengths 8-14: the range `research/language-capability/RESULTS.md` reports
    # 1.000 on.  `research/inference-cost` sampled lengths up to 22, where the
    # program is outside its own validated range and disagrees with `balanced`
    # on 5 of 12 episodes -- its `out/inproc.json` records `all_agree: false`
    # while its RESULTS.md sec 4.1 claims 12/12.  Reproduced and reported, but
    # arm D has to be an equivalence reference here, so the cases stay in range.
    eps = LC.dataset(200, seed0=100000, split='test')
    eps = [e for e in eps if 8 <= e['length'] <= 14][:12]
    cases = [{'text': e['text'].decoded, '_string': e['string'], '_label': e['label']} for e in eps]

    def reference(c):
        """The same job written directly, off the same typed observation.

        `text` is `(length, 128 prompt bytes)`; the lesson's single template puts
        the string at byte 14 and terminates it with `.`.  This is the function
        the search was set to find, reading exactly what the program reads.
        """
        _n, b = c['text']
        d = 0
        for i in range(14, 128):
            ch = b[i]
            if ch == 40:
                d += 1
            elif ch == 41:
                d -= 1
                if d < 0:
                    return {'answer': False}
            else:
                break
        return {'answer': d == 0}

    return {'name': 'language', 'program': program, 'registry': registry, 'cases': cases,
            'reference': reference, 'tolerance': 0.0, 'coerce': bool,
            'what': 'grammaticality of a Dyck word from a 128-byte prompt'}


# ---------------------------------------------------------------------------
# visual -- 32x32 RGB screenshot -> widget rectangles, 64,346 operations
# ---------------------------------------------------------------------------
def visual(seeds=(200, 201, 202)):
    _use('visual-ladder')
    from common import FLAT, Registry, bytes_type, episode, load
    import rung3_widgets as R

    found = load('rung3')
    registry = Registry()
    probe = episode(0, 'train', **FLAT)
    W, H = probe['width'], probe['height']
    offsets = R.offset_pool(W)
    p0 = R.same_scaffold(registry, W, H)
    m0 = registry.register_module(p0.harden(found['s0']['chosen']))
    p1 = R.corner_scaffold(registry, W, H, m0, offsets)
    m1 = registry.register_module(p1.harden(found['s1']['chosen']))
    p2 = R.rect_scaffold(registry, W, H, m0, offsets)
    m2 = registry.register_module(p2.harden(found['s2']['chosen']))

    eps = [episode(s, 'test', **FLAT) for s in seeds]
    ep = eps[0]
    BT = bytes_type(ep['width'], ep['height'])
    program = R.assembly(registry, BT, R.interior_positions(ep), m1, m2)
    cases = [{'observation': tuple(e['pixels']), '_w': e['width'], '_h': e['height']} for e in eps]

    def reference(c):
        """The same parse, written directly as a nested loop over the raster.

        Corner: the pixel differs from its left AND its upper neighbour.
        Extent: the length of the contiguous same-colour run right / down.
        Keys  : the pixel's own packed RGB and its left neighbour's.
        """
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

    return {'name': 'visual', 'program': program, 'registry': registry, 'cases': cases,
            'reference': reference, 'tolerance': 0.0,
            'what': '32x32 RGB screenshot -> set of widget rectangles with parent keys'}


# ---------------------------------------------------------------------------
# computer -- the frozen agent program only; the OS round trip is measured apart
# ---------------------------------------------------------------------------
def _zero(t):
    """A legal value of `t` with every scalar at zero, or at its lower bound.

    The frozen agent program's *cost* is a function of the declared widths its
    types fix, not of the content, so a synthetic observation of the declared
    type prices the program path without a live kernel.  The kernel round trip
    is not modelled here at all -- it is 967.85 ms and it is measured in
    `research/inference-cost/RESULTS.md` sec 2.
    """
    if t.kind == 'bool':
        return False
    if t.kind == 'tuple':
        return tuple(_zero(x) for x in t.items)
    if t.kind == 'set':
        return frozenset()
    if t.bounds is not None and not t.bounds[0] <= 0 <= t.bounds[1]:
        return t.bounds[0]
    return 0


def computer():
    """The frozen agent program, loaded from the artifact its own track froze.

    Program only.  There is no arm D: the hand-written reference in
    `research/inference-cost/refs.py` decides between three action templates from
    a parsed document rather than from this program's typed observation, so it is
    not the same function of the same input and would not be an honest arm D.
    """
    from tcn.runtime import load_program
    program, registry = load_program(ROOT / 'research/computer-capability/out/agent_program.json')
    types = dict(program.inputs)
    base = {k: _zero(t) for k, t in types.items()}
    # a length of zero drives `pos := sub(length, 1)` below its declared bound
    # (0, 4096) and the program raises -- correctly -- before it does any work,
    # so the synthetic observations carry a plausible non-empty document
    text = b'note 3\n'
    cases = []
    for digit in (48, 51, 57):
        c = dict(base)
        body = text[:-2] + bytes([digit]) + text[-1:]
        buf = list(base['terminal'][1])
        buf[:len(body)] = list(body)
        c['terminal'] = (len(body), tuple(buf))
        cases.append(c)
    return {'name': 'computer', 'program': program, 'registry': registry, 'cases': cases,
            'reference': None, 'tolerance': 0.0,
            'what': 'frozen computer-use agent program (program only, synthetic observation)'}


FIXTURES = {'mixed': mixed, 'language': language, 'visual': visual, 'computer': computer}
